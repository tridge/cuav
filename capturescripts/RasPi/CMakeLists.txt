cmake_minimum_required(VERSION 2.8)

project(cuavraw)

include_directories("/opt/vc/include")
include_directories("./")

link_directories("/opt/vc/lib")

SET(CMAKE_CURRENT_SOURCE_DIR ./)

SET(COMPILE_DEFINITIONS -Werror)

set (COMMON_SOURCES
   RaspiCamControl.c
   RaspiCLI.c
   RaspiPreview.c)

set (MMAL_LIBS mmal_core mmal_util mmal_vc_client)

add_executable(cuavraw ${COMMON_SOURCES} cuavraw.c  RaspiTex.c RaspiTexUtil.c cuav_util.c )
target_link_libraries(cuavraw ${MMAL_LIBS} vcos bcm_host brcmGLESv2 brcmEGL jpeg pthread vcsm m)
