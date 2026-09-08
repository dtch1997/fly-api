# headless rendering env for Track B (no sudo: OSMesa extracted from ubuntu deb)
# recipe: apt-get download libosmesa6 libglapi-mesa && dpkg -x <deb> prefix
export OSMESA_PREFIX=${OSMESA_PREFIX:-/tmp/claude-2038/-mnt-nw-home-d-tan-jarvis-monorepo-jarvis-os/7e623888-45f1-4fd0-9563-5048d259e964/scratchpad/osmesa-local/prefix}
export LD_LIBRARY_PATH=$OSMESA_PREFIX/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
