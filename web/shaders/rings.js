// Ring sprites. Giant Fiber marker (steady, breathes with its activity) and input flash (expands and fades).
import { ACTIVITY_GLSL } from "./common.js";

export const gfRingVertex = /* glsl */ `
${ACTIVITY_GLSL}
attribute vec3 aLayered;
attribute float aIndex;
uniform float uMix;
uniform float uSizeScale;
uniform float uWorldSize;
uniform float uLayersScale;  // the reticle grows in the layers view, where the GF is otherwise lost in its slab
uniform float uMaxPoint;
varying float vAct;
void main() {
  vec3 p = mix(position, aLayered, uMix);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  vec4 a = activityAt(aIndex);
  vAct = max(a.r, a.g);
  gl_PointSize = clamp(uWorldSize * mix(1.0, uLayersScale, uMix) * (1.0 + 0.25 * vAct) * uSizeScale / -mv.z, 6.0, 1.6 * uMaxPoint);
}
`;

export const gfRingFragment = /* glsl */ `
uniform vec3 uColor;
uniform float uTime;
varying float vAct;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float w = 0.06;
  float ring = smoothstep(w, 0.0, abs(d - 0.78));
  // four hairline ticks at the cardinal points: a reticle rather than a halo
  vec2 c = gl_PointCoord - 0.5;
  float tick = step(0.86, d) * step(d, 0.98) * max(step(abs(c.x), 0.012), step(abs(c.y), 0.012));
  float breathe = 0.85 + 0.15 * sin(uTime * 1.3);
  float a = (ring * 0.22 + tick * 0.35) * breathe * (0.35 + 0.9 * vAct);
  gl_FragColor = vec4(uColor * a, 1.0);
}
`;

export const flashVertex = /* glsl */ `
attribute vec3 aLayered;
attribute float aStart;
uniform float uMix;
uniform float uTime;
uniform float uDuration;
uniform float uSizeScale;
uniform float uWorldSize;
varying float vK;
void main() {
  vec3 p = mix(position, aLayered, uMix);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float k = (uTime - aStart) / uDuration;
  vK = k;
  gl_PointSize = (k < 0.0 || k > 1.0) ? 0.0 : clamp(uWorldSize * uSizeScale / -mv.z, 8.0, 512.0);
}
`;

export const flashFragment = /* glsl */ `
uniform vec3 uColor;
varying float vK;
void main() {
  if (vK < 0.0 || vK > 1.0) discard;
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float e = 1.0 - pow(1.0 - vK, 3.0);          // ease-out radius
  float r = 0.1 + 0.88 * e;
  float w = 0.008 + 0.03 * (1.0 - vK);         // a hairline by the time it is large
  float ring = smoothstep(w, 0.0, abs(d - r));
  float inner = smoothstep(r, 0.0, d) * 0.05 * (1.0 - vK);
  float a = (ring * 0.55 + inner) * pow(1.0 - vK, 2.0);
  gl_FragColor = vec4(uColor * a, 1.0);
}
`;
