#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

static constexpr int RN = 16384; // anneau d'échantillons
static constexpr int FN = 8192;  // taille FFT
static constexpr int NB = FN / 2;
static constexpr int HB = 160;   // histogramme 0,5 dB de -70 à +10 LUFS
using Spec = std::array<float, NB>;

static juce::String T (const char* s) { return juce::String::fromUTF8 (s); }
static size_t histIndex (float lufs) { return (size_t) juce::jlimit (0, HB - 1, (int) ((lufs + 70.f) * 2.f)); }
static float histCentre (size_t i) { return -70.f + ((float) i + 0.5f) / 2.f; }

// interpolation Catmull-Rom : plus grande valeur entre p1 et p2 (vrai pic approché, x4)
static float interSamplePeak (float p0, float p1, float p2, float p3)
{
    float m = 0.f;
    for (float t : { 0.25f, 0.5f, 0.75f })
    {
        const float t2 = t * t, t3 = t2 * t;
        const float v = 0.5f * ((2.f * p1) + (-p0 + p2) * t + (2.f * p0 - 5.f * p1 + 4.f * p2 - p3) * t2
                                + (-p0 + 3.f * p1 - 3.f * p2 + p3) * t3);
        m = std::max (m, std::abs (v));
    }
    return m;
}

//==============================================================================
class ScopeProcessor : public juce::AudioProcessor
{
public:
    ScopeProcessor()
        : AudioProcessor (BusesProperties()
                              .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                              .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
    {
        resetMeters();
    }

    const juce::String getName() const override { return "Scope Master"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}
    void releaseResources() override {}
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;

    bool isBusesLayoutSupported (const BusesLayout& l) const override
    {
        return l.getMainInputChannelSet() == l.getMainOutputChannelSet()
               && l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
    }

    void prepareToPlay (double sr, int) override
    {
        binLen = juce::jmax (1, (int) (sr * 0.1));
        for (int c = 0; c < 2; ++c)
        {
            hs[c].coefficients = juce::dsp::IIR::Coefficients<float>::makeHighShelf (sr, 1681.0f, 0.7071f, 1.585f);
            hp[c].coefficients = juce::dsp::IIR::Coefficients<float>::makeHighPass (sr, 38.0f, 0.5f);
            hs[c].reset();
            hp[c].reset();
            for (int k = 0; k < 3; ++k) z[c][k] = 0.f;
        }
        kAcc = 0;
        rawAcc = 0;
        kCnt = 0;
    }

    void processBlock (juce::AudioBuffer<float>& b, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals nd;
        const int n = b.getNumSamples();
        if (b.getNumChannels() < 1) return;
        const float* L = b.getReadPointer (0);
        const float* R = b.getNumChannels() > 1 ? b.getReadPointer (1) : L;
        int w = wr.load (std::memory_order_relaxed);
        float pl = pkL.load(), pr = pkR.load(), tl = tpL.load(), tr = tpR.load();

        for (int i = 0; i < n; ++i)
        {
            const float l = L[i], r = R[i];
            bufL[(size_t) w] = l;
            bufR[(size_t) w] = r;
            w = (w + 1) & (RN - 1);

            pl = std::max (pl, std::abs (l));
            pr = std::max (pr, std::abs (r));
            tl = std::max ({ tl, std::abs (l), interSamplePeak (z[0][0], z[0][1], z[0][2], l) });
            tr = std::max ({ tr, std::abs (r), interSamplePeak (z[1][0], z[1][1], z[1][2], r) });
            z[0][0] = z[0][1]; z[0][1] = z[0][2]; z[0][2] = l;
            z[1][0] = z[1][1]; z[1][1] = z[1][2]; z[1][2] = r;

            const float kl = hp[0].processSample (hs[0].processSample (l));
            const float kr = hp[1].processSample (hs[1].processSample (r));
            kAcc += (double) kl * kl + (double) kr * kr;
            rawAcc += (double) l * l + (double) r * r;
            if (++kCnt >= binLen) endBin();
        }
        wr.store (w, std::memory_order_release);
        pkL.store (pl);
        pkR.store (pr);
        tpL.store (tl);
        tpR.store (tr);
    }

    void resetMeters()
    {
        pkL.store (0.f);
        pkR.store (0.f);
        tpL.store (0.f);
        tpR.store (0.f);
        lufsM.store (-100.f);
        lufsS.store (-100.f);
        rmsDb.store (-100.f);
        for (auto& h : histI) h.store (0);
        for (auto& h : histS) h.store (0);
    }

    static float toLufs (double e) { return e > 1e-12 ? (float) (-0.691 + 10.0 * std::log10 (e)) : -100.f; }

    std::array<float, RN> bufL {}, bufR {};
    std::atomic<int> wr { 0 };
    std::atomic<float> pkL { 0.f }, pkR { 0.f }, tpL { 0.f }, tpR { 0.f },
        lufsM { -100.f }, lufsS { -100.f }, rmsDb { -100.f };
    std::array<std::atomic<int>, HB> histI, histS;

private:
    void endBin()
    {
        const size_t idx = (size_t) (bi & 63);
        bins[idx] = kAcc / (double) kCnt;
        raws[idx] = rawAcc / (2.0 * (double) kCnt);
        ++bi;
        kAcc = 0;
        rawAcc = 0;
        kCnt = 0;

        if (bi >= 4)
        {
            double m = 0;
            for (int j = 1; j <= 4; ++j) m += bins[(size_t) ((bi - j) & 63)];
            const float lm = toLufs (m / 4.0);
            lufsM.store (lm);
            if (lm > -70.f) histI[histIndex (lm)].fetch_add (1);
        }
        if (bi >= 30)
        {
            double s = 0, rw = 0;
            for (int j = 1; j <= 30; ++j)
            {
                s += bins[(size_t) ((bi - j) & 63)];
                rw += raws[(size_t) ((bi - j) & 63)];
            }
            const float ls = toLufs (s / 30.0);
            lufsS.store (ls);
            rmsDb.store ((float) (10.0 * std::log10 (rw / 30.0 + 1e-12)));
            if (ls > -70.f) histS[histIndex (ls)].fetch_add (1);
        }
    }

    juce::dsp::IIR::Filter<float> hs[2], hp[2];
    std::array<double, 64> bins {}, raws {};
    float z[2][3] {};
    double kAcc = 0, rawAcc = 0;
    int kCnt = 0, binLen = 4800, bi = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ScopeProcessor)
};

//==============================================================================
class ScopeEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit ScopeEditor (ScopeProcessor& pr)
        : AudioProcessorEditor (&pr), p (pr), sgImg (juce::Image::ARGB, 900, 160, true)
    {
        spec.fill (-120.f);
        peaks.fill (-120.f);
        for (auto& v : bandDb) v = -120.f;
        setResizable (true, true);
        setResizeLimits (720, 480, 1800, 1100);
        setSize (1000, 720);
        startTimerHz (30);
    }
    ~ScopeEditor() override { stopTimer(); }

    void mouseDown (const juce::MouseEvent& e) override
    {
        if (rSpec.contains (e.position))
        {
            slopeIdx = (slopeIdx + 1) % 3;
            return;
        }
        p.resetMeters();
        peaks.fill (-120.f);
        hist.clear();
        sgImg.clear (sgImg.getBounds());
    }

    void resized() override
    {
        auto a = getLocalBounds().toFloat().reduced (12.f);
        const float gap = 10.f;
        rSpec = a.removeFromTop (a.getHeight() * 0.40f);
        a.removeFromTop (gap);
        rSg = a.removeFromTop (a.getHeight() * 0.30f);
        a.removeFromTop (gap);
        const float gs = juce::jmin (a.getHeight(), a.getWidth() * 0.38f);
        rGon = a.removeFromLeft (gs).removeFromTop (gs);
        a.removeFromLeft (gap);
        rSt = a.removeFromTop (50.f);
        a.removeFromTop (gap * 0.5f);
        rHist = a.removeFromTop (a.getHeight() * 0.42f);
        a.removeFromTop (gap * 0.5f);
        rRead = a;
    }

    void timerCallback() override
    {
        const int w = p.wr.load (std::memory_order_acquire);
        const bool active = (w != lastW);
        lastW = w;

        for (int i = 0; i < FN; ++i)
        {
            const size_t k = (size_t) (((unsigned) (w - FN + i)) & (unsigned) (RN - 1));
            fftData[(size_t) i] = 0.5f * (p.bufL[k] + p.bufR[k]);
        }
        std::fill (fftData.begin() + FN, fftData.end(), 0.f);
        win.multiplyWithWindowingTable (fftData.data(), FN);
        fft.performFrequencyOnlyForwardTransform (fftData.data());
        for (size_t i = 0; i < NB; ++i)
        {
            const float d = juce::Decibels::gainToDecibels (fftData[i] / (FN * 0.25f), -120.f);
            spec[i] = spec[i] * 0.65f + d * 0.35f;
            peaks[i] = std::max (d, peaks[i] - 0.35f);
        }

        // énergie par zone
        const float sr = srNow();
        for (size_t b = 0; b < 6; ++b)
        {
            const int k0 = juce::jlimit (0, NB - 1, (int) (kEdges[b] * FN / sr));
            const int k1 = juce::jlimit (k0, NB - 1, (int) (kEdges[b + 1] * FN / sr));
            double s = 0;
            for (int k = k0; k <= k1; ++k) s += std::pow (10.0, spec[(size_t) k] / 10.0);
            bandDb[b] = (float) (10.0 * std::log10 (s / (double) (k1 - k0 + 1) + 1e-12));
        }

        // image stéréo sur les 2048 derniers échantillons
        double ll = 0, rr = 0, lr = 0, mm = 0, ss = 0;
        for (int i = 0; i < 2048; ++i)
        {
            const size_t k = (size_t) (((unsigned) (w - 2048 + i)) & (unsigned) (RN - 1));
            const double l = p.bufL[k], q = p.bufR[k], m = (l + q) / 2, s = (l - q) / 2;
            ll += l * l; rr += q * q; lr += l * q; mm += m * m; ss += s * s;
        }
        if (active && ll + rr > 1e-8)
        {
            corr += (float) (lr / std::sqrt (ll * rr + 1e-12) - corr) * 0.15f;
            sidePct += (float) (ss / (mm + ss + 1e-12) * 100.0 - sidePct) * 0.15f;
            balDb = (float) (10.0 * std::log10 ((ll + 1e-12) / (rr + 1e-12)));
        }

        ++tick;
        if (active)
        {
            pushSpectrogramColumn();
            if (tick % 3 == 0)
            {
                hist.push_back (p.lufsS.load());
                if (hist.size() > 600) hist.erase (hist.begin());
            }
        }
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (bg);
        drawSpectrum (g, rSpec);
        drawSpectrogram (g, rSg);
        drawGonio (g, rGon);
        drawStereo (g, rSt);
        drawHist (g, rHist);
        drawReadouts (g, rRead);
    }

private:
    const juce::Colour bg { 0xff15111f }, panel { 0xff1c1730 }, line { 0xff322a52 }, ink { 0xffece8ff },
        dim { 0xff9a92c4 }, pink { 0xffff4fa3 }, cyan { 0xff6fd3ff }, amber { 0xffffc857 };

    static constexpr float kEdges[7] { 20.f, 60.f, 250.f, 500.f, 2000.f, 6000.f, 20000.f };
    float srNow() const { return p.getSampleRate() > 8000.0 ? (float) p.getSampleRate() : 44100.f; }
    float slope() const { static const float s[3] { 3.f, 4.5f, 0.f }; return s[slopeIdx]; }
    static juce::String fmt (float v, int d = 1) { return v > -99.f ? juce::String (v, d) : juce::String ("--"); }
    float xOf (float f, const juce::Rectangle<float>& r) const { return r.getX() + std::log (f / 20.f) / std::log (1000.f) * r.getWidth(); }

    static juce::Colour heat (float t)
    {
        t = juce::jlimit (0.f, 1.f, t);
        return juce::Colour::fromFloatRGBA (0.08f + 0.92f * juce::jmin (1.f, t * 1.5f),
                                            0.05f + 0.95f * juce::jlimit (0.f, 1.f, (t - 0.4f) * 1.7f),
                                            0.14f + 0.35f * std::sin (t * 3.14159f), 1.f);
    }

    void pushSpectrogramColumn()
    {
        juce::Image::BitmapData bd (sgImg, juce::Image::BitmapData::readWrite);
        const int W = bd.width, H = bd.height;
        const float sr = srNow();
        for (int y = 0; y < H; ++y)
        {
            auto* row = bd.getLinePointer (y);
            std::memmove (row, row + bd.pixelStride, (size_t) (W - 1) * (size_t) bd.pixelStride);
            const float f = 20.f * std::pow (1000.f, 1.f - (float) y / (float) H);
            const int b = juce::jlimit (0, NB - 1, (int) (f * FN / sr));
            const float v = spec[(size_t) b] + slope() * std::log2 (f / 1000.f);
            bd.setPixelColour (W - 1, y, heat ((v + 90.f) / 70.f));
        }
    }

    void drawSpectrum (juce::Graphics& g, juce::Rectangle<float> r)
    {
        g.setColour (panel);
        g.fillRoundedRectangle (r, 10.f);
        juce::Graphics::ScopedSaveState ss (g);
        g.reduceClipRegion (r.toNearestInt());
        g.setFont (11.f);

        for (float f : { 50.f, 100.f, 200.f, 500.f, 1000.f, 2000.f, 5000.f, 10000.f })
        {
            const float x = xOf (f, r);
            g.setColour (line);
            g.drawVerticalLine ((int) x, r.getY(), r.getBottom());
            g.setColour (dim);
            g.drawText (f >= 1000.f ? juce::String (f / 1000.f, 0) + "k" : juce::String ((int) f),
                        (int) x + 3, (int) r.getBottom() - 16, 40, 14, juce::Justification::left);
        }
        for (int d = -20; d >= -80; d -= 20)
        {
            const float y = r.getY() + (float) (-d) / 90.f * r.getHeight();
            g.setColour (line);
            g.drawHorizontalLine ((int) y, r.getX(), r.getRight());
            g.setColour (dim);
            g.drawText (juce::String (d) + " dB", (int) r.getX() + 4, (int) y - 14, 50, 12, juce::Justification::left);
        }

        // zones de fréquences avec niveau
        const char* names[6] { "Sub", "Basses", "Bas-méd.", "Médiums", "Présence", "Air" };
        for (int b = 0; b < 6; ++b)
        {
            const float x0 = xOf (kEdges[b], r), x1 = xOf (kEdges[b + 1], r);
            g.setColour (dim);
            g.drawText (T (names[b]) + "  " + fmt (bandDb[(size_t) b], 0), (int) x0, (int) r.getY() + 4,
                        (int) (x1 - x0), 14, juce::Justification::centred);
        }
        g.setColour (dim);
        g.drawText (T ("Pente +") + juce::String (slope(), 1) + " dB/oct (clic : changer)", (int) r.getRight() - 230,
                    (int) r.getY() + 20, 224, 12, juce::Justification::right);

        const float sr = srNow();
        auto curve = [&] (const Spec& a, juce::Path& path)
        {
            bool first = true;
            for (float px = 0.f; px <= r.getWidth(); px += 2.f)
            {
                const float f = 20.f * std::pow (1000.f, px / r.getWidth());
                const int b0 = juce::jlimit (0, NB - 1, (int) (f * FN / sr));
                const int b1 = juce::jlimit (b0, NB - 1, (int) std::ceil (f * 1.03f * FN / sr));
                float v = -120.f;
                for (int b = b0; b <= b1; ++b) v = std::max (v, a[(size_t) b]);
                v += slope() * std::log2 (f / 1000.f);
                const float y = r.getY() + juce::jlimit (0.f, 1.f, -v / 90.f) * r.getHeight();
                if (first) { path.startNewSubPath (r.getX() + px, y); first = false; }
                else path.lineTo (r.getX() + px, y);
            }
        };

        juce::Path ps, pp;
        curve (spec, ps);
        curve (peaks, pp);
        juce::Path fill (ps);
        fill.lineTo (r.getRight(), r.getBottom());
        fill.lineTo (r.getX(), r.getBottom());
        fill.closeSubPath();
        g.setColour (pink.withAlpha (0.3f));
        g.fillPath (fill);
        g.setColour (pink);
        g.strokePath (ps, juce::PathStrokeType (1.6f));
        g.setColour (amber);
        g.strokePath (pp, juce::PathStrokeType (1.f));
    }

    void drawSpectrogram (juce::Graphics& g, juce::Rectangle<float> r)
    {
        g.setColour (panel);
        g.fillRoundedRectangle (r, 10.f);
        juce::Graphics::ScopedSaveState ss (g);
        g.reduceClipRegion (r.toNearestInt());
        g.setImageResamplingQuality (juce::Graphics::mediumResamplingQuality);
        g.drawImage (sgImg, r.reduced (1.f), juce::RectanglePlacement::stretchToFit);
        g.setColour (dim);
        g.setFont (11.f);
        g.drawText (T ("Spectrogramme"), (int) r.getX() + 6, (int) r.getY() + 4, 120, 14, juce::Justification::left);
    }

    void drawGonio (juce::Graphics& g, juce::Rectangle<float> r)
    {
        g.setColour (panel);
        g.fillRoundedRectangle (r, 10.f);
        juce::Graphics::ScopedSaveState ss (g);
        g.reduceClipRegion (r.toNearestInt());
        const auto c = r.getCentre();
        g.setColour (line);
        g.drawLine (c.x, r.getY(), c.x, r.getBottom());
        g.drawLine (r.getX(), c.y, r.getRight(), c.y);
        g.drawLine (r.getX(), r.getY(), r.getRight(), r.getBottom());
        g.drawLine (r.getRight(), r.getY(), r.getX(), r.getBottom());
        g.setColour (dim);
        g.setFont (12.f);
        g.drawText ("G", (int) r.getX() + 6, (int) r.getY() + 4, 20, 14, juce::Justification::left);
        g.drawText ("D", (int) r.getRight() - 26, (int) r.getY() + 4, 20, 14, juce::Justification::right);
        g.drawText ("M", (int) c.x + 5, (int) r.getY() + 4, 20, 14, juce::Justification::left);

        const int w = p.wr.load (std::memory_order_acquire);
        const int M = 1024;
        float pk = 0.05f;
        for (int i = 0; i < M; ++i)
        {
            const size_t k = (size_t) (((unsigned) (w - M + i)) & (unsigned) (RN - 1));
            pk = std::max ({ pk, std::abs (p.bufL[k]), std::abs (p.bufR[k]) });
        }
        gs += (juce::jmin (4.f, 1.f / pk) - gs) * 0.1f;
        const float k0 = r.getWidth() * 0.22f * gs;
        for (int i = 0; i < M; ++i)
        {
            const size_t k = (size_t) (((unsigned) (w - M + i)) & (unsigned) (RN - 1));
            const float l = p.bufL[k], q = p.bufR[k];
            g.setColour (pink.withAlpha (0.15f + 0.85f * (float) i / (float) M));
            g.fillRect (c.x + (q - l) * k0, c.y - (l + q) * k0, 1.7f, 1.7f);
        }
    }

    void meterBar (juce::Graphics& g, juce::Rectangle<float> r, const juce::String& label, const juce::String& value,
                   float pos, juce::Colour c0, juce::Colour c1)
    {
        g.setFont (11.f);
        g.setColour (dim);
        g.drawText (label, r.removeFromLeft (r.getWidth() * 0.26f).toNearestInt(), juce::Justification::centredLeft);
        g.setColour (ink);
        g.drawText (value, r.removeFromRight (56.f).toNearestInt(), juce::Justification::centredRight);
        auto tr = r.reduced (4.f, r.getHeight() * 0.32f);
        g.setGradientFill (juce::ColourGradient (c0, tr.getX(), 0.f, c1, tr.getRight(), 0.f, false));
        g.fillRoundedRectangle (tr, tr.getHeight() / 2.f);
        const float x = tr.getX() + juce::jlimit (0.f, 1.f, pos) * tr.getWidth();
        g.setColour (ink);
        g.fillRoundedRectangle (x - 2.f, r.getY() + 2.f, 4.f, r.getHeight() - 4.f, 2.f);
    }

    void drawStereo (juce::Graphics& g, juce::Rectangle<float> r)
    {
        g.setColour (panel);
        g.fillRoundedRectangle (r, 10.f);
        auto in = r.reduced (8.f, 4.f);
        meterBar (g, in.removeFromTop (in.getHeight() / 2.f), T ("Corrélation"), juce::String (corr, 2),
                  (corr + 1.f) / 2.f, pink, cyan);
        meterBar (g, in, T ("Largeur (Side)"), juce::String (sidePct, 0) + " %", sidePct / 100.f, cyan, pink);
    }

    void drawHist (juce::Graphics& g, juce::Rectangle<float> r)
    {
        g.setColour (panel);
        g.fillRoundedRectangle (r, 10.f);
        juce::Graphics::ScopedSaveState ss (g);
        g.reduceClipRegion (r.toNearestInt());
        auto Y = [&] (float v) { return r.getY() + juce::jlimit (0.f, 1.f, (-6.f - v) / 34.f) * r.getHeight(); };
        g.setFont (10.f);
        for (float v : { -10.f, -20.f, -30.f })
        {
            g.setColour (line);
            g.drawHorizontalLine ((int) Y (v), r.getX(), r.getRight());
            g.setColour (dim);
            g.drawText (juce::String ((int) v), (int) r.getX() + 4, (int) Y (v) - 12, 30, 11, juce::Justification::left);
        }
        const float dash[2] { 6.f, 5.f };
        g.setColour (amber);
        g.drawDashedLine (juce::Line<float> (r.getX(), Y (-14.f), r.getRight(), Y (-14.f)), dash, 2, 1.f);
        g.drawText (T ("-14 diffusion"), (int) r.getRight() - 96, (int) Y (-14.f) - 12, 90, 11, juce::Justification::right);
        g.setColour (dim);
        g.drawText (T ("Sonie court terme, 60 s"), (int) r.getX() + 34, (int) r.getY() + 3, 160, 11, juce::Justification::left);

        juce::Path path;
        bool first = true;
        const int n = (int) hist.size();
        for (int i = 0; i < n; ++i)
        {
            const float x = r.getRight() - (float) (n - 1 - i) / 600.f * r.getWidth();
            const float y = Y (hist[(size_t) i] > -99.f ? hist[(size_t) i] : -40.f);
            if (first) { path.startNewSubPath (x, y); first = false; }
            else path.lineTo (x, y);
        }
        g.setColour (cyan);
        g.strokePath (path, juce::PathStrokeType (2.f));
    }

    float integrated() const
    {
        double s = 0;
        long n = 0;
        std::array<int, HB> h {};
        for (size_t i = 0; i < HB; ++i) h[i] = p.histI[i].load();
        auto mean = [&] (float gate)
        {
            s = 0;
            n = 0;
            for (size_t i = 0; i < HB; ++i)
                if (histCentre (i) > gate) { s += h[i] * std::pow (10.0, (histCentre (i) + 0.691) / 10.0); n += h[i]; }
            return n ? (float) (10.0 * std::log10 (s / (double) n) - 0.691) : -100.f;
        };
        const float g1 = mean (-71.f);
        return g1 > -99.f ? mean (g1 - 10.f) : -100.f;
    }

    float lra() const
    {
        std::array<int, HB> h {};
        double s = 0;
        long n = 0;
        for (size_t i = 0; i < HB; ++i)
        {
            h[i] = p.histS[i].load();
            s += h[i] * std::pow (10.0, (histCentre (i) + 0.691) / 10.0);
            n += h[i];
        }
        if (n < 20) return -100.f;
        const float gate = (float) (10.0 * std::log10 (s / (double) n) - 0.691) - 20.f;
        long tot = 0;
        for (size_t i = 0; i < HB; ++i)
            if (histCentre (i) > gate) tot += h[i];
        if (tot < 2) return -100.f;
        auto pct = [&] (double q) -> float
        {
            const long target = (long) (q * (double) tot);
            long c = 0;
            for (size_t i = 0; i < HB; ++i)
                if (histCentre (i) > gate)
                {
                    c += h[i];
                    if (c >= target) return histCentre (i);
                }
            return histCentre (HB - 1);
        };
        return pct (0.95) - pct (0.10);
    }

    void cell (juce::Graphics& g, juce::Rectangle<float> r, const juce::String& v, const juce::String& label, bool hot = false)
    {
        g.setColour (hot ? pink : ink);
        g.setFont (19.f);
        g.drawText (v, r.removeFromTop (r.getHeight() * 0.62f), juce::Justification::bottomLeft);
        g.setColour (dim);
        g.setFont (11.f);
        g.drawText (label, r, juce::Justification::topLeft);
    }

    void drawReadouts (juce::Graphics& g, juce::Rectangle<float> r)
    {
        const float I = integrated(), L = lra();
        const float pl = juce::Decibels::gainToDecibels (p.pkL.load(), -100.f);
        const float pr = juce::Decibels::gainToDecibels (p.pkR.load(), -100.f);
        const float tp = juce::Decibels::gainToDecibels (std::max (p.tpL.load(), p.tpR.load()), -100.f);
        const float rms = p.rmsDb.load();
        const float spk = std::max (pl, pr);

        auto signedStr = [] (float v) { return (v >= 0.f ? juce::String ("+") : juce::String()) + juce::String (v, 1); };
        const juce::String vals[12] {
            fmt (p.lufsM.load()), fmt (p.lufsS.load()), fmt (I), fmt (L),
            fmt (pl), fmt (pr), fmt (tp), I > -99.f ? signedStr (I + 14.f) : juce::String ("--"),
            fmt (rms), (tp > -99.f && I > -99.f) ? fmt (tp - I) : juce::String ("--"),
            (spk > -99.f && rms > -99.f) ? fmt (spk - rms) : juce::String ("--"), fmt (balDb) };
        const char* labels[12] { "Momentané (LUFS)", "Court terme (LUFS)", "Intégré (LUFS)", "LRA (LU)",
                                 "Crête G (dBFS)", "Crête D (dBFS)", "True peak (dBTP)", "Écart à -14 (LU)",
                                 "RMS 3 s (dBFS)", "PLR (dB)", "Facteur de crête (dB)", "Balance G/D (dB)" };
        const float cw = r.getWidth() / 4.f, ch = r.getHeight() / 3.f;
        for (int i = 0; i < 12; ++i)
        {
            const bool hot = (i == 4 && pl > -0.3f) || (i == 5 && pr > -0.3f) || (i == 6 && tp > -1.0f);
            cell (g, juce::Rectangle<float> (r.getX() + (float) (i % 4) * cw, r.getY() + (float) (i / 4) * ch, cw, ch).reduced (4.f, 1.f),
                  vals[i], T (labels[i]), hot);
        }
    }

    ScopeProcessor& p;
    juce::Image sgImg;
    juce::dsp::FFT fft { 13 };
    juce::dsp::WindowingFunction<float> win { (size_t) FN, juce::dsp::WindowingFunction<float>::hann, false };
    std::array<float, FN * 2> fftData {};
    Spec spec {}, peaks {};
    std::array<float, 6> bandDb {};
    std::vector<float> hist;
    juce::Rectangle<float> rSpec, rSg, rGon, rSt, rHist, rRead;
    float gs = 1.f, corr = 1.f, sidePct = 0.f, balDb = -100.f;
    int slopeIdx = 0, lastW = 0, tick = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ScopeEditor)
};


juce::AudioProcessorEditor* ScopeProcessor::createEditor() { return new ScopeEditor (*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new ScopeProcessor(); }
