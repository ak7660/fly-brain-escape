// Synapse lines with a travelling pulse (bright head, soft tail) from pre (t=0) to post (t=1).
// Pulse intensity = presynaptic activity × normalized |weight|; color follows the presynaptic sign.
import { ACTIVITY_GLSL, HASH_GLSL } from "./common.js";

export const synapseVertex = /* glsl */ `
${ACTIVITY_GLSL}
${HASH_GLSL}
attribute vec3 aLayered;
attribute vec4 aEdge;      // x: t (0 pre, 1 post) · y: pre index · z: sign · w: normalized |val|
uniform float uMix;
uniform float uIdleAlpha;  // 0 → silent edges are collapsed out of the viewport (no overdraw)
uniform vec3 uExc;
uniform vec3 uInh;
uniform vec3 uMod;
varying float vT;
varying float vDrive;
varying float vW;
varying float vOffset;
varying vec3 vColor;
void main() {
  vec4 a = activityAt(aEdge.y);
  float act = max(a.r, a.g);                  // a decision highlight also sends pulses down the DN's axon
  vDrive = smoothstep(0.05, 0.5, act) * aEdge.w;
  if (vDrive <= 0.0 && uIdleAlpha <= 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // both ends outside the clip volume: the segment is culled
    return;
  }
  vec3 p = mix(position, aLayered, uMix);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  vT = aEdge.x;
  vW = aEdge.w;
  vOffset = hash11(aEdge.y);
  vColor = aEdge.z > 0.5 ? uExc : (aEdge.z < -0.5 ? uInh : uMod);
}
`;

export const synapseFragment = /* glsl */ `
uniform float uTime;
uniform float uSpeed;      // pulses per second along one edge
uniform float uIdleAlpha;
uniform float uGain;
uniform float uFade;
uniform float uTail;       // tail decay: larger = shorter dash
uniform float uBase;       // steady glow of an active edge, relative to the pulse head
varying float vT;
varying float vDrive;
varying float vW;
varying float vOffset;
varying vec3 vColor;
void main() {
  float head = fract(uTime * uSpeed + vOffset);
  float x = fract(head - vT);                          // distance behind the head, along the edge
  float pulse = smoothstep(0.0, 0.01, x) * exp(-x * uTail);
  float lit = vDrive * (uBase + pulse);
  float a = (uIdleAlpha * (0.35 + 0.65 * vW) + lit * uGain) * uFade;
  gl_FragColor = vec4(vColor * a, 1.0);
}
`;
