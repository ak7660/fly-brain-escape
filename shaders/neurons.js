// Neuron sprites: soft round points whose size and brightness follow activity. Additive, no depth write.
// Activity texel: R = change-from-rest activity 0..1, G = decision highlight 0..1 (brightness/size/white core, no hue).
import { ACTIVITY_GLSL } from "./common.js";

export const neuronVertex = /* glsl */ `
${ACTIVITY_GLSL}
attribute vec3 aLayered;
attribute vec3 aColor;
attribute float aSize;
attribute float aIndex;
attribute float aIdle;      // per-neuron idle brightness multiplier (inhibitory 0.6, DN 1.3)
uniform float uMix;
uniform float uSizeScale;   // viewport px / (2 tan(fov/2)): world µm → px at depth 1
uniform float uMaxPoint;    // sprite clamp in px, scaled with the viewport
uniform float uIdle;        // idle brightness
uniform float uCamDist;     // camera → orbit target distance
uniform float uDepthRange;  // µm over which brightness fades with depth
varying vec3 vColor;
varying float vGlow;
varying float vHot;
void main() {
  vec3 p = mix(position, aLayered, uMix);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  vec4 a = activityAt(aIndex);
  float act = a.r;
  float hi = a.g;
  float e = max(act, hi);
  float s = aSize * (1.0 + 1.25 * sqrt(e) + 0.8 * hi);
  gl_PointSize = clamp(s * uSizeScale / -mv.z, 1.5, uMaxPoint);
  float depth = clamp(1.0 - 0.3 * (-mv.z - uCamDist) / uDepthRange, 0.6, 1.2);
  vColor = aColor;
  vGlow = (uIdle * aIdle + (1.0 - uIdle * aIdle) * e) * depth;
  vHot = max(act * act, hi);
}
`;

export const neuronFragment = /* glsl */ `
uniform float uGain;
varying vec3 vColor;
varying float vGlow;
varying float vHot;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c) * 2.0;
  if (d > 1.0) discard;
  float halo = pow(1.0 - d, 2.2);
  float core = smoothstep(0.42, 0.0, d);
  // idle: a soft dim dot; active: a hot, slightly whitened core that feeds the bloom
  vec3 col = vColor * (halo * 0.7 + core * 0.6) * vGlow * uGain;
  col += vec3(1.0) * core * vHot * 0.45 * uGain;
  gl_FragColor = vec4(col, 1.0);
}
`;
