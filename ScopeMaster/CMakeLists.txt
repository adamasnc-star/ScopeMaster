cmake_minimum_required(VERSION 3.22)
project(ScopeMaster VERSION 1.1.0)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(JUCE
  GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
  GIT_TAG 8.0.12
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(ScopeMaster
  COMPANY_NAME "Maison"
  BUNDLE_ID com.maison.scopemaster
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  PLUGIN_MANUFACTURER_CODE Mais
  PLUGIN_CODE Scpm
  FORMATS VST3
  VST3_CATEGORIES Fx Analyzer
  PRODUCT_NAME "Scope Master"
  COPY_PLUGIN_AFTER_BUILD FALSE)

target_sources(ScopeMaster PRIVATE Plugin.cpp)
target_compile_definitions(ScopeMaster PUBLIC
  JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(ScopeMaster
  PRIVATE juce::juce_audio_utils juce::juce_dsp
  PUBLIC juce::juce_recommended_config_flags)
