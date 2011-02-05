#!/bin/sh

../prefix/bin/fgfs \
    --disable-intro-music \
    --airport=YKRY \
    --units-meters \
    --lat=-26.5779682 \
    --lon=151.8392086 \
    --heading=170 \
    --altitude=0 \
    --geometry=600x400 \
    --jpg-httpd=5502 \
    --bpp=32 \
    --disable-anti-alias-hud \
    --disable-hud-3d \
    --disable-enhanced-lighting \
    --disable-distance-attenuation \
    --disable-horizon-effect \
    --shading-flat \
    --disable-textures \
    --aircraft=Rascal110-JSBSim \
    --generic=socket,out,50,,5501,udp,MAVLink \
    --generic=socket,in,50,,5500,udp,MAVLink \
    --timeofday=noon \
    --fdm=jsb \
    --disable-sound \
    --disable-fullscreen \
    --disable-random-objects \
    --disable-ai-models \
    --shading-flat \
    --fog-disable \
    --disable-specular-highlight \
    --disable-skyblend \
    --fg-scenery="/home/tridge/project/UAV/scenery" \
    --atlas=socket,out,1,localhost,5100,udp \
    --disable-anti-alias-hud \
    --wind=0@0 \
    $*


# --control=mouse \
#    --wind=0@0 \
#    --heading=0 \
#    --turbulence=0.0 \
#    --geometry=400x300 \
#   --disable-panel \
#    --disable-horizon-effect \
#    --disable-clouds \
#   --prop:/sim/frame-rate-throttle-hz=30 \
  