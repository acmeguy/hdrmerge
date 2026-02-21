cmake_minimum_required(VERSION 3.10)

# Generate compile_commands.json for IDEs, linters and language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_policy(SET CMP0025 NEW)

project(hdrmerge)

if(WIN32)
    if(NOT(ALGLIB_ROOT))
        message(FATAL_ERROR "When building on Windows, you need to download 'ALGLIB' source and define its root directory in ALGLIB_ROOT !")
    else()
        message(STATUS "ALGLIB_ROOT = ${ALGLIB_ROOT}")
    endif()
endif()

if(APPLE)
    if(NOT(ALGLIB_ROOT))
        message(FATAL_ERROR "Passage of ALGLIB_ROOT to cmake failed. Download ALGLIB and define its root directory in ALGLIB_ROOT")
    else()
        message(STATUS "ALGLIB_ROOT = ${ALGLIB_ROOT}")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unknown-pragmas")
set(PLATFORM "")

# Required and optional packages
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# Generate resources automatically
set(CMAKE_AUTORCC ON)

find_package(Qt5 5.6 REQUIRED Core Widgets)

#include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

find_package(LibRaw REQUIRED)
find_package(Exiv2 REQUIRED)
# libdeflate for fast DEFLATE compression in DNG writer
find_library(LIBDEFLATE_LIBRARY deflate PATHS /opt/homebrew/lib)
find_path(LIBDEFLATE_INCLUDE_DIR libdeflate.h PATHS /opt/homebrew/include)
if(LIBDEFLATE_LIBRARY AND LIBDEFLATE_INCLUDE_DIR)
    message(STATUS "Found libdeflate: ${LIBDEFLATE_LIBRARY}")
    add_definitions(-DHAVE_LIBDEFLATE)
else()
    message(STATUS "libdeflate not found, falling back to zlib")
endif()
# libjxl for JPEG XL compression in DNG 1.7 (16-bit float only)
find_library(LIBJXL_LIBRARY jxl PATHS /opt/homebrew/lib)
find_library(LIBJXL_THREADS_LIBRARY jxl_threads PATHS /opt/homebrew/lib)
find_path(LIBJXL_INCLUDE_DIR jxl/encode.h PATHS /opt/homebrew/include)
if(LIBJXL_LIBRARY AND LIBJXL_INCLUDE_DIR)
    message(STATUS "Found libjxl: ${LIBJXL_LIBRARY}")
    add_definitions(-DHAVE_LIBJXL)
else()
    message(STATUS "libjxl not found, JPEG XL DNG compression disabled")
endif()
find_package(ZLIB REQUIRED)
find_package(OpenMP)

# Optional: OpenCV for feature-based alignment (--align-features)
find_package(OpenCV QUIET COMPONENTS core features2d calib3d imgproc)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}: feature-based alignment available")
    add_definitions(-DHAVE_OPENCV)
else()
    message(STATUS "OpenCV not found: feature-based alignment disabled (MTB only)")
endif()

# Optional: ONNX Runtime for adaptive tone curves (--auto-curves)
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS /opt/homebrew/lib)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /opt/homebrew/include/onnxruntime /opt/homebrew/include)
if(ONNXRUNTIME_LIBRARY AND ONNXRUNTIME_INCLUDE_DIR)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
    add_definitions(-DHAVE_ONNXRUNTIME)
    include_directories("${ONNXRUNTIME_INCLUDE_DIR}")
else()
    message(STATUS "ONNX Runtime not found: --auto-curves disabled")
endif()

# Commented-out as it doesn't link
#find_package(Boost 1.46 COMPONENTS unit_test_framework)

if(WIN32 OR APPLE)
    set(ALGLIB_INCLUDES ${ALGLIB_ROOT}/src)
else()
    find_package(ALGLIB REQUIRED)
endif()

include_directories(
    "${LibRaw_INCLUDE_DIR}"
    "${EXIV2_INCLUDE_DIR}"
    "${ZLIB_INCLUDE_DIRS}"
    "${ALGLIB_INCLUDES}"
)
if(LIBDEFLATE_LIBRARY AND LIBDEFLATE_INCLUDE_DIR)
    include_directories("${LIBDEFLATE_INCLUDE_DIR}")
endif()
if(LIBJXL_LIBRARY AND LIBJXL_INCLUDE_DIR)
    include_directories("${LIBJXL_INCLUDE_DIR}")
endif()

if(NOT(CMAKE_BUILD_TYPE))
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    if(CMAKE_COMPILER_IS_GNUCC)
        set(STRIP -s)
    endif()
endif()

# The version number.
set(HDRMERGE_VERSION_MAJOR 0)
set(HDRMERGE_VERSION_MINOR 5)
set(HDRMERGE_VERSION_REV 0)
set(HDRMERGE_VERSION ${HDRMERGE_VERSION_MAJOR}.${HDRMERGE_VERSION_MINOR}.${HDRMERGE_VERSION_REV})

# configure a header file to pass some of the CMake settings to the source code
configure_file(
    "${PROJECT_SOURCE_DIR}/src/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h"
)

# add the binary tree to the search path for include files so that we will find config.h
include_directories("${PROJECT_BINARY_DIR}")

set(PLATFORM_SOURCES "")
if(WIN32)
    # This is required by libraw.h
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWIN32 -DLIBRAW_NODLL")
    set(PLATFORM WIN32)
    configure_file(
        "${PROJECT_SOURCE_DIR}/data/winres.rc"
        "${PROJECT_BINARY_DIR}/winres.rc"
    )
    set(PLATFORM_SOURCES "${PROJECT_BINARY_DIR}/winres.rc")
elseif(APPLE)
    set(PLATFORM MACOSX_BUNDLE)
    set(MACOSX_BUNDLE_ICON_FILE "icon")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "org.hdrmerge")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING "HDRMerge for macOS v${HDRMERGE_VERSION}")
    set(MACOSX_BUNDLE_BUNDLE_NAME "HDRMerge")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${HDRMERGE_VERSION}")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "${HDRMERGE_VERSION}")
    set(MACOSX_BUNDLE_INFO_COPYRIGHT "Copyright 2018 Javier Celaya")
    set(PLATFORM_SOURCES "data/images/icon.icns")
endif()

# Sources and headers
set(hdrmerge_sources
    src/Image.cpp
    src/ImageStack.cpp
    src/Bitmap.cpp
    src/RawParameters.cpp
    src/EditableMask.cpp
    src/DngFloatWriter.cpp
    src/TiffDirectory.cpp
    src/BoxBlur.cpp
    src/ExifTransfer.cpp
    src/ImageIO.cpp
    src/AdaptiveCurves.cpp
    src/Resizer.cpp
)

set(hdrmerge_gui_sources
    src/AboutDialog.cpp
    src/MainWindow.cpp
    src/PreviewWidget.cpp
    src/DraggableScrollArea.cpp
    src/DngPropertiesDialog.cpp
    src/LoadOptionsDialog.cpp
    src/FileSystem.cpp
)

set(hdrmerge_qobject_headers
    src/MainWindow.hpp
    src/PreviewWidget.hpp
    src/LoadOptionsDialog.hpp
    src/DraggableScrollArea.hpp
    src/DngPropertiesDialog.hpp
    src/AboutDialog.hpp
    src/FileSystem.hpp
)

set(hdrmerge_translations
    data/hdrmerge_es.ts
)

if(WIN32)
    set(alglib_sources
        ${ALGLIB_ROOT}/src/alglibinternal.cpp
        ${ALGLIB_ROOT}/src/alglibmisc.cpp
        ${ALGLIB_ROOT}/src/ap.cpp
        ${ALGLIB_ROOT}/src/dataanalysis.cpp
        ${ALGLIB_ROOT}/src/diffequations.cpp
        ${ALGLIB_ROOT}/src/fasttransforms.cpp
        ${ALGLIB_ROOT}/src/integration.cpp
        ${ALGLIB_ROOT}/src/interpolation.cpp
        ${ALGLIB_ROOT}/src/linalg.cpp
        ${ALGLIB_ROOT}/src/optimization.cpp
        ${ALGLIB_ROOT}/src/solvers.cpp
        ${ALGLIB_ROOT}/src/specialfunctions.cpp
        ${ALGLIB_ROOT}/src/statistics.cpp
    )
endif()

if(APPLE)
    set(alglib_sources
        ${ALGLIB_ROOT}/src/alglibinternal.cpp
        ${ALGLIB_ROOT}/src/alglibmisc.cpp
        ${ALGLIB_ROOT}/src/ap.cpp
        ${ALGLIB_ROOT}/src/dataanalysis.cpp
        ${ALGLIB_ROOT}/src/diffequations.cpp
        ${ALGLIB_ROOT}/src/fasttransforms.cpp
        ${ALGLIB_ROOT}/src/integration.cpp
        ${ALGLIB_ROOT}/src/interpolation.cpp
        ${ALGLIB_ROOT}/src/linalg.cpp
        ${ALGLIB_ROOT}/src/optimization.cpp
        ${ALGLIB_ROOT}/src/solvers.cpp
        ${ALGLIB_ROOT}/src/specialfunctions.cpp
        ${ALGLIB_ROOT}/src/statistics.cpp
    )
endif()

# Libs
set(hdrmerge_libs
    "${LibRaw_r_LIBRARIES}"
    "${EXIV2_LIBRARY}"
    "${ZLIB_LIBRARIES}"
)

if(WIN32)
    get_filename_component(LIB_PATH "${EXIV2_LIBRARY}" PATH)
    find_library(EXPAT_LIBRARY libexpat.a "${LIB_PATH}")
    find_library(ICONV_LIBRARY libiconv.a "${LIB_PATH}")
    find_library(GETTEXT_LIBRARY libintl.a "${LIB_PATH}")
    set(hdrmerge_libs
        ${hdrmerge_libs}
        "${EXPAT_LIBRARY}"
        "${GETTEXT_LIBRARY}"
        "${ICONV_LIBRARY}"
        psapi
        wsock32
        imm32
        winmm
    )
    remove_definitions(-DQT_DLL)
else()
    set(hdrmerge_libs
        "${hdrmerge_libs}"
        "${ALGLIB_LIBRARIES}"
    )
endif()

#QT4_ADD_TRANSLATION(hdrmerge_qm ${hdrmerge_translations})

# Generate the XML version of hdrmerge_qm
foreach(file ${hdrmerge_qm})
    get_filename_component(basename ${file} NAME)
    set(HDRMERGE_QM_XML "${HDRMERGE_QM_XML}
    <file>${basename}</file>")
endforeach()

configure_file(
    "${PROJECT_SOURCE_DIR}/data/translations.qrc.in"
    "${PROJECT_BINARY_DIR}/translations.qrc"
)

if(WIN32)
    add_library(alglib-objects OBJECT ${alglib_sources})
endif()

if(APPLE)
    add_library(alglib-objects OBJECT ${alglib_sources})
endif()

if(WIN32)
    add_executable(hdrmerge ${PLATFORM}
        src/main.cpp
        src/Launcher.cpp
        data/resources.qrc
        ${hdrmerge_translations}
        ${hdrmerge_sources}
        ${hdrmerge_gui_sources}
        ${hdrmerge_moc}
        $<TARGET_OBJECTS:alglib-objects>
        "${PLATFORM_SOURCES}"
    )
elseif(APPLE)
    add_executable(hdrmerge ${PLATFORM}
        src/main.cpp
        src/Launcher.cpp
        data/resources.qrc
        $<TARGET_OBJECTS:alglib-objects>
        ${hdrmerge_translations}
        ${hdrmerge_sources}
        ${hdrmerge_gui_sources}
        ${hdrmerge_moc}
        "${PLATFORM_SOURCES}"
    )
else()
    add_executable(hdrmerge ${PLATFORM}
        src/main.cpp
        src/Launcher.cpp
        data/resources.qrc
        ${hdrmerge_translations}
        ${hdrmerge_sources}
        ${hdrmerge_gui_sources}
        ${hdrmerge_moc}
        "${PLATFORM_SOURCES}"
    )
endif()

if(OpenCV_FOUND)
    set(hdrmerge_libs ${hdrmerge_libs} ${OpenCV_LIBS})
endif()
if(LIBDEFLATE_LIBRARY)
    set(hdrmerge_libs ${hdrmerge_libs} ${LIBDEFLATE_LIBRARY})
endif()
if(LIBJXL_LIBRARY)
    set(hdrmerge_libs ${hdrmerge_libs} ${LIBJXL_LIBRARY})
    if(LIBJXL_THREADS_LIBRARY)
        set(hdrmerge_libs ${hdrmerge_libs} ${LIBJXL_THREADS_LIBRARY})
    endif()
endif()
if(ONNXRUNTIME_LIBRARY)
    set(hdrmerge_libs ${hdrmerge_libs} ${ONNXRUNTIME_LIBRARY})
endif()
target_link_libraries(hdrmerge ${STRIP} ${hdrmerge_libs} Qt5::Widgets)

if(WIN32)
    # Compile a target without GUI, for the .com executable
    add_executable(hdrmerge-nogui
        src/main.cpp
        src/Launcher.cpp
        data/resources.qrc
        ${hdrmerge_translations}
        ${hdrmerge_sources}
        ${hdrmerge_moc}
        $<TARGET_OBJECTS:alglib-objects>
        "${PLATFORM_SOURCES}"
    )
    
    # WARNING: ImageIO class need QImage which need Qt5::Widgets
    #          Launcher.cpp need QApplication which need Qt5::Widgets
    target_link_libraries(hdrmerge-nogui ${STRIP} ${hdrmerge_libs} Qt5::Widgets Qt5::Core)
    set_target_properties(hdrmerge-nogui PROPERTIES COMPILE_DEFINITIONS "NO_GUI")
    # Create the installer with makensis
    find_program(MAKENSIS_EXECUTABLE makensis.exe PATH_SUFFIXES "NSIS/Bin")
    if(MAKENSIS_EXECUTABLE)
        message(STATUS "NSIS found")
        set(SETUP_PROG hdrmerge-setup${WIN_ARCH}-${HDRMERGE_VERSION}.exe)
        string(REPLACE "/" "\\" PROJ_SRC_DIR "${PROJECT_SOURCE_DIR}")
        message(STATUS "PROJ_SRC_DIR       = ${PROJ_SRC_DIR}")

        # ASSUMING YOU'RE USING EXIV2 BUNDLED WITH MSYS2
        find_program(MINGW_LIB_DIR exiv2.exe)
        if(MINGW_LIB_DIR)
            get_filename_component(MINGW_LIB_DIR "${MINGW_LIB_DIR}" PATH)
            string(REPLACE "/" "\\" MINGW_LIB_DIR "${MINGW_LIB_DIR}")
            message(STATUS "MINGW_LIB_DIR      = ${MINGW_LIB_DIR}")

            set(QT5_PLUGINS_DIR "${MINGW_LIB_DIR}\\..\\share\\qt5\\plugins")
            message(STATUS "QT5_PLUGINS_DIR    = ${QT5_PLUGINS_DIR}")

            configure_file(
                "${PROJECT_SOURCE_DIR}/data/setup.nsi"
                "${PROJECT_BINARY_DIR}/setup.nsi" @ONLY
            )
            add_custom_command(OUTPUT ${SETUP_PROG}
                COMMAND "${MAKENSIS_EXECUTABLE}" -V2 setup.nsi
                MAIN_DEPENDENCY setup.nsi
                DEPENDS "${PROJECT_BINARY_DIR}/hdrmerge.exe"
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
            add_custom_target(hdrmerge-setup ALL DEPENDS "${PROJECT_BINARY_DIR}/${SETUP_PROG}")
        else(MINGW_LIB_DIR)
            message(STATUS "Unable to locate the 'bin' directory to get the dependencies")
        endif(MINGW_LIB_DIR)
    else()
        message(STATUS "NSIS not found")
    endif()
elseif(APPLE)
    set_source_files_properties(data/images/icon.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
    install(TARGETS hdrmerge BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR})
    # Bundle libjxl and its transitive dependencies into the .app
    if(LIBJXL_LIBRARY AND LIBJXL_INCLUDE_DIR)
        set(BUNDLE_FW "$<TARGET_BUNDLE_DIR:hdrmerge>/Contents/Frameworks")
        set(BUNDLE_EXE "$<TARGET_BUNDLE_DIR:hdrmerge>/Contents/MacOS/hdrmerge")
        # Resolve symlinks to get the real versioned dylib paths
        get_filename_component(LIBJXL_REAL "${LIBJXL_LIBRARY}" REALPATH)
        get_filename_component(LIBJXL_THREADS_REAL "${LIBJXL_THREADS_LIBRARY}" REALPATH)
        # Find transitive dependencies
        get_filename_component(_jxl_lib_dir "${LIBJXL_REAL}" DIRECTORY)
        find_library(LIBJXL_CMS_LIBRARY jxl_cms PATHS "${_jxl_lib_dir}" NO_DEFAULT_PATH)
        find_library(LIBHWY_LIBRARY hwy PATHS /opt/homebrew/lib)
        find_library(LIBBROTLIDEC_LIBRARY brotlidec PATHS /opt/homebrew/lib)
        find_library(LIBBROTLIENC_LIBRARY brotlienc PATHS /opt/homebrew/lib)
        find_library(LIBBROTLICOMMON_LIBRARY brotlicommon PATHS /opt/homebrew/lib)
        find_library(LIBLCMS2_LIBRARY lcms2 PATHS /opt/homebrew/lib)
        set(_jxl_bundle_libs
            "${LIBJXL_REAL}"
            "${LIBJXL_THREADS_REAL}"
        )
        foreach(_dep LIBJXL_CMS_LIBRARY LIBHWY_LIBRARY LIBBROTLIDEC_LIBRARY
                     LIBBROTLIENC_LIBRARY LIBBROTLICOMMON_LIBRARY LIBLCMS2_LIBRARY)
            if(${_dep})
                get_filename_component(_real "${${_dep}}" REALPATH)
                list(APPEND _jxl_bundle_libs "${_real}")
            endif()
        endforeach()
        # Post-build: copy dylibs and rewrite load paths
        add_custom_command(TARGET hdrmerge POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_FW}"
            COMMENT "Bundling libjxl dylibs into app bundle"
        )
        foreach(_lib ${_jxl_bundle_libs})
            get_filename_component(_name "${_lib}" NAME)
            add_custom_command(TARGET hdrmerge POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_lib}" "${BUNDLE_FW}/${_name}"
                COMMAND chmod u+w "${BUNDLE_FW}/${_name}"
                COMMAND install_name_tool -id "@executable_path/../Frameworks/${_name}" "${BUNDLE_FW}/${_name}"
            )
        endforeach()
        # Rewrite load paths in the main executable and in each bundled dylib
        add_custom_command(TARGET hdrmerge POST_BUILD
            COMMAND ${CMAKE_SOURCE_DIR}/cmake/fix_jxl_rpaths.sh "${BUNDLE_EXE}" "${BUNDLE_FW}"
            COMMENT "Fixing libjxl load paths"
        )
    endif()
else()
    if(NOT DEFINED CMAKE_INSTALL_BINDIR)
        set(CMAKE_INSTALL_BINDIR "bin")
    endif()
    message(STATUS "Install binary to: " ${CMAKE_INSTALL_BINDIR})
    install(TARGETS hdrmerge DESTINATION ${CMAKE_INSTALL_BINDIR})
    find_program(XDG-DESKTOP-MENU_EXECUTABLE xdg-desktop-menu)
    find_program(XDG-ICON-RESOURCE_EXECUTABLE xdg-icon-resource)
    if(XDG-DESKTOP-MENU_EXECUTABLE AND XDG-ICON-RESOURCE_EXECUTABLE)
        configure_file(
            "${PROJECT_SOURCE_DIR}/data/hdrmerge.desktop"
            "${PROJECT_BINARY_DIR}/hdrmerge.desktop"
        )
        install(CODE "
            execute_process(COMMAND \"${XDG-ICON-RESOURCE_EXECUTABLE}\" install --novendor --size 128 \"${PROJECT_SOURCE_DIR}/data/images/icon.png\" hdrmerge-icon)
            execute_process(COMMAND \"${XDG-DESKTOP-MENU_EXECUTABLE}\" install --novendor \"${PROJECT_BINARY_DIR}/hdrmerge.desktop\")"
        )
    endif()
endif()

if(Boost_FOUND)
    add_subdirectory(test)
endif()

