// Shared GLSL. Activity lives in a DataTexture (R = display activity 0..1, G = decision highlight 0..1)
// laid out row-major, one texel per neuron, sampled by vertex index with texelFetch (WebGL2).
export const ACTIVITY_GLSL = /* glsl */ `
uniform sampler2D uAct;
uniform float uActWidth;
vec4 activityAt(float idx) {
  float x = mod(idx, uActWidth);
  float y = floor(idx / uActWidth);
  return texelFetch(uAct, ivec2(int(x), int(y)), 0);
}
`;

export const HASH_GLSL = /* glsl */ `
float hash11(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
`;
