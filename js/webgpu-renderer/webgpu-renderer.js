// Copyright 2020 Brandon Jones
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// This import installs hooks that help us output better formatted shader errors
import './wgsl-debug-helper.js';

import { Renderer } from '../renderer.js';
import { ProjectionUniformsSize, ViewUniformsSize, BIND_GROUP } from './shaders/common.js';
import { PBRRenderBundleHelper, PBRClusteredRenderBundleHelper } from './pbr-render-bundle-helper.js';
import { DepthVisualization, DepthSliceVisualization, ClusterDistanceVisualization, LightsPerClusterVisualization } from './debug-visualizations.js';
import { LightSpriteVertexSource, LightSpriteFragmentSource } from './shaders/light-sprite.js';
import { vec2, vec3, vec4, mat4 } from '../third-party/gl-matrix/dist/esm/index.js';
import { WebGPUTextureLoader } from '../third-party/web-texture-tool/build/webgpu-texture-loader.js';

import { ClusterBoundsSource, ClusterLightsSource, DISPATCH_SIZE, TOTAL_TILES, CLUSTER_LIGHTS_SIZE } from './shaders/clustered-compute.js';

const SAMPLE_COUNT = 4;
const DEPTH_FORMAT = "depth24plus";

// Can reuse these for every PBR material
const materialUniforms = new Float32Array(4 + 4 + 4);
const baseColorFactor = new Float32Array(materialUniforms.buffer, 0, 4);
const metallicRoughnessFactor = new Float32Array(materialUniforms.buffer, 4 * 4, 2);
const emissiveFactor = new Float32Array(materialUniforms.buffer, 8 * 4, 3);

const emptyArray = new Uint32Array(1);

export class WebGPURenderer extends Renderer {
  constructor() {
    super();

    this.context = this.canvas.getContext('webgpu');

    this.outputHelpers = {
      'naive-forward': PBRRenderBundleHelper,
      'clustered-forward': PBRClusteredRenderBundleHelper,
      'depth': DepthVisualization,
      'depth-slice': DepthSliceVisualization,
      'cluster-distance': ClusterDistanceVisualization,
      'lights-per-cluster': LightsPerClusterVisualization,
    };

    // WebXR state
    this.xrSession = null;
    this.xrBinding = null;
    this.xrLayer = null;
    this.xrRefSpace = null;
    this.xrBaseRefSpace = null;
    this.xrDepthTextures = [];
    this.xrViewBuffers = [];
    this.xrPrevTime = 0;

    // Light selection state
    this.selectedLightIndex = -1;
    this.releasingLightIndex = -1; // For color lerp back
    this.wasTransientPointerActive = false;
    this.selectionStartDistance = 0; // XZ distance from ray origin to light
    this.selectionStartRayDirY = 0; // Ray Y direction at selection start
    this.selectionStartLightY = 0; // Light Y at selection start
    this.originalLightColor = vec3.create();
    this.releasingLightColor = vec3.create();
    this.colorAmplify = 1.0; // Amplification factor during selection

    // Two-handed color control
    this.secondHandActive = false;
    this.secondHandStartX = 0;
    this.colorRamp = [
      vec3.fromValues(1, 0.2, 0.2),   // Red
      vec3.fromValues(1, 0.5, 0.1),   // Orange
      vec3.fromValues(1, 1, 0.2),     // Yellow
      vec3.fromValues(0.2, 1, 0.2),   // Green
      vec3.fromValues(0.2, 0.8, 1),   // Cyan
      vec3.fromValues(0.3, 0.3, 1),   // Blue
      vec3.fromValues(0.8, 0.2, 1),   // Purple
      vec3.fromValues(1, 0.4, 0.8),   // Pink
    ];

    // Teleportation state
    this.teleportTarget = null;
    this.teleportStartPos = null;
    this.teleportStartRayOrigin = null;
    this.playerOffset = vec3.fromValues(0, 0, 0);
    this.isTeleporting = false; // true = teleport mode, false = light mode
  }

  async init() {
    this.outputRenderBundles = {};

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
      xrCompatible: true
    });

    // Enable compressed textures if available
    const requiredFeatures = [];
    if (this.adapter.features.has('texture-compression-bc')) {
      requiredFeatures.push('texture-compression-bc');
    }

    if (this.adapter.features.has('texture-compression-etc2')) {
      requiredFeatures.push('texture-compression-etc2');
    }

    if (this.adapter.features.has('texture-compression-astc')) {
      requiredFeatures.push('texture-compression-astc');
    }

    this.device = await this.adapter.requestDevice({requiredFeatures});

    this.contextFormat = 'bgra8unorm';
    if (navigator.gpu.getPreferredCanvasFormat) {
      this.contextFormat = navigator.gpu.getPreferredCanvasFormat();
    } else if (this.context.getPreferredFormat) {
      this.contextFormat = this.context.getPreferredFormat(this.adapter);
    }

    this.context.configure({
      device: this.device,
      format: this.contextFormat,
      alphaMode: "opaque",
    });

    this.renderBundleDescriptor = {
      colorFormats: [ this.contextFormat ],
      depthStencilFormat: DEPTH_FORMAT,
      sampleCount: SAMPLE_COUNT
    };

    // XR render bundle descriptor (no MSAA)
    this.xrRenderBundleDescriptor = {
      colorFormats: [ this.contextFormat ],
      depthStencilFormat: DEPTH_FORMAT,
      sampleCount: 1
    };

    // Just for debugging my shader helper stuff. This is expected to fail.
    /*this.device.createShaderModule({
      label: 'Test Shader',
      code: `
        // 頂点シェーダー
        @vertex
        fn main(@location(0) inPosition : vec3) -> @builtin(position) vec4<f32> {
          return vec3<f32>(inPosition, 1.0);
        }
      `
    });*/

    this.textureLoader = new WebGPUTextureLoader(this.device);

    this.colorAttachment = {
      // view is acquired and set in onResize.
      view: undefined,
      // renderTarget is acquired and set in onFrame.
      resolveTarget: undefined,
      loadOp: 'clear',
      clearValue: { r: 0.0, g: 0.0, b: 0.5, a: 1.0 },
      storeOp: 'discard',
    };

    this.depthAttachment = {
      // view is acquired and set in onResize.
      view: undefined,
      depthLoadOp: 'clear',
      depthClearValue: 1.0,
      depthStoreOp: 'discard',
    };

    this.renderPassDescriptor = {
      colorAttachments: [this.colorAttachment],
      depthStencilAttachment: this.depthAttachment
    };

    this.bindGroupLayouts = {
      frame: this.device.createBindGroupLayout({
        label: `frame-bgl`,
        entries: [{
          binding: 0, // Projection uniforms
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: {},
        }, {
          binding: 1, // View uniforms
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
          buffer: {}
        }, {
          binding: 2, // Light uniforms
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' }
        }, {
          binding: 3, // Cluster Lights storage
          visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        }]
      }),

      material: this.device.createBindGroupLayout({
        label: `material-bgl`,
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {}
        },
        {
          binding: 1, // defaultSampler
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {}
        },
        {
          binding: 2, // baseColorTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        },
        {
          binding: 3, // normalTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        },
        {
          binding: 4, // metallicRoughnessTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        },
        {
          binding: 5, // occlusionTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        },
        {
          binding: 6, // emissiveTexture
          visibility: GPUShaderStage.FRAGMENT,
          texture: {}
        }]
      }),

      primitive: this.device.createBindGroupLayout({
        label: `primitive-bgl`,
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: {}
        }]
      }),

      cluster: this.device.createBindGroupLayout({
        label: `cluster-bgl`,
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' }
        }]
      }),
    };

    this.pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [
        this.bindGroupLayouts.frame, // set 0
        this.bindGroupLayouts.material, // set 1
        this.bindGroupLayouts.primitive, // set 2
      ]
    });

    this.projectionBuffer = this.device.createBuffer({
      size: ProjectionUniformsSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });

    this.viewBuffer = this.device.createBuffer({
      size: ViewUniformsSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });

    // Per-eye view buffers for XR stereo rendering
    this.xrViewBuffers = [
      this.device.createBuffer({ size: ViewUniformsSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM }),
      this.device.createBuffer({ size: ViewUniformsSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM }),
    ];
    this.xrViewData = new Float32Array(ViewUniformsSize / 4);

    // Per-eye projection buffers for XR (projection matrices differ per eye)
    this.xrProjectionBuffers = [
      this.device.createBuffer({ size: ProjectionUniformsSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM }),
      this.device.createBuffer({ size: ProjectionUniformsSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM }),
    ];
    this.xrProjectionData = new Float32Array(ProjectionUniformsSize / 4);
    this.xrInverseProjection = mat4.create();

    this.lightsBuffer = this.device.createBuffer({
      size: this.lightManager.uniformArray.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });

    this.clusterLightsBuffer = this.device.createBuffer({
      size: CLUSTER_LIGHTS_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.bindGroups = {
      frame: this.device.createBindGroup({
        layout: this.bindGroupLayouts.frame,
        entries: [{
          binding: 0,
          resource: {
            buffer: this.projectionBuffer,
          },
        }, {
          binding: 1,
          resource: {
            buffer: this.viewBuffer,
          },
        }, {
          binding: 2,
          resource: {
            buffer: this.lightsBuffer,
          },
        }, {
          binding: 3,
          resource: {
            buffer: this.clusterLightsBuffer
          }
        }],
      })
    };

    // Per-eye frame bind groups for XR stereo rendering
    this.xrFrameBindGroups = this.xrViewBuffers.map((viewBuffer, i) =>
      this.device.createBindGroup({
        layout: this.bindGroupLayouts.frame,
        entries: [{
          binding: 0,
          resource: { buffer: this.xrProjectionBuffers[i] },
        }, {
          binding: 1,
          resource: { buffer: viewBuffer },
        }, {
          binding: 2,
          resource: { buffer: this.lightsBuffer },
        }, {
          binding: 3,
          resource: { buffer: this.clusterLightsBuffer }
        }],
      })
    );

    this.blackTextureView = this.textureLoader.fromColor(0, 0, 0, 0).texture.createView();
    this.whiteTextureView = this.textureLoader.fromColor(1.0, 1.0, 1.0, 1.0).texture.createView();
    this.blueTextureView = this.textureLoader.fromColor(0, 0, 1.0, 0).texture.createView();

    // Setup a render pipeline for drawing the light sprites
    const lightSpriteVertexModule = this.device.createShaderModule({
      code: LightSpriteVertexSource,
      label: 'Light Sprite'
    });
    const lightSpriteFragmentModule = this.device.createShaderModule({
      code: LightSpriteFragmentSource,
      label: 'Light Sprite'
    });
    const lightSpritePipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayouts.frame]
    });

    this.lightSpritePipeline = this.device.createRenderPipeline({
      label: `light-sprite-pipeline`,
      layout: lightSpritePipelineLayout,
      vertex: {
        module: lightSpriteVertexModule,
        entryPoint: 'vertexMain'
      },
      fragment: {
        module: lightSpriteFragmentModule,
        entryPoint: 'fragmentMain',
        targets: [{
          format: this.contextFormat,
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one',
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one",
            },
          },
        }],
      },
      primitive: {
        topology: 'triangle-strip',
        stripIndexFormat: 'uint32'
      },
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: 'less',
        format: DEPTH_FORMAT,
      },
      multisample: {
        count: SAMPLE_COUNT,
      }
    });

    // Teleport reticle pipeline (XR only, no MSAA)
    const reticleShaderModule = this.device.createShaderModule({
      label: 'Teleport Reticle',
      code: `
        struct Uniforms {
          viewProj: mat4x4<f32>,
          position: vec3<f32>,
          radius: f32,
        };
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vertexMain(@builtin(vertex_index) idx: u32) -> VertexOutput {
          var pos = array<vec2<f32>, 4>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>(1.0, -1.0),
            vec2<f32>(-1.0, 1.0),
            vec2<f32>(1.0, 1.0)
          );
          var output: VertexOutput;
          output.uv = pos[idx];
          let worldPos = vec3<f32>(
            uniforms.position.x + pos[idx].x * uniforms.radius,
            uniforms.position.y + 0.01,
            uniforms.position.z + pos[idx].y * uniforms.radius
          );
          output.position = uniforms.viewProj * vec4<f32>(worldPos, 1.0);
          return output;
        }

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
          let dist = length(uv);
          let ring = smoothstep(0.7, 0.75, dist) * (1.0 - smoothstep(0.85, 0.9, dist));
          let center = 1.0 - smoothstep(0.0, 0.15, dist);
          let alpha = (ring + center * 0.5) * 0.8;
          if (alpha < 0.01) { discard; }
          return vec4<f32>(0.2, 0.8, 1.0, alpha);
        }
      `
    });

    this.reticleBindGroupLayout = this.device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {}
      }]
    });

    // Per-eye reticle buffers (to avoid writeBuffer timing issues in stereo)
    this.reticleUniformBuffers = [];
    this.reticleBindGroups = [];
    for (let i = 0; i < 2; i++) {
      const buffer = this.device.createBuffer({
        size: 80, // mat4 + vec3 + f32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      this.reticleUniformBuffers.push(buffer);
      this.reticleBindGroups.push(this.device.createBindGroup({
        layout: this.reticleBindGroupLayout,
        entries: [{
          binding: 0,
          resource: { buffer }
        }]
      }));
    }

    this.reticlePipeline = this.device.createRenderPipeline({
      label: 'reticle-pipeline',
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.reticleBindGroupLayout]
      }),
      vertex: {
        module: reticleShaderModule,
        entryPoint: 'vertexMain'
      },
      fragment: {
        module: reticleShaderModule,
        entryPoint: 'fragmentMain',
        targets: [{
          format: this.contextFormat,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one' }
          }
        }]
      },
      primitive: {
        topology: 'triangle-strip',
        stripIndexFormat: 'uint32'
      },
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: 'less',
        format: DEPTH_FORMAT,
      },
      multisample: { count: 1 }
    });

    this.reticleUniformData = new Float32Array(20); // mat4 + vec3 + f32

    // XR light sprite pipeline (no MSAA)
    this.xrLightSpritePipeline = this.device.createRenderPipeline({
      label: `xr-light-sprite-pipeline`,
      layout: lightSpritePipelineLayout,
      vertex: {
        module: lightSpriteVertexModule,
        entryPoint: 'vertexMain'
      },
      fragment: {
        module: lightSpriteFragmentModule,
        entryPoint: 'fragmentMain',
        targets: [{
          format: this.contextFormat,
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one',
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one",
            },
          },
        }],
      },
      primitive: {
        topology: 'triangle-strip',
        stripIndexFormat: 'uint32'
      },
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: 'less',
        format: DEPTH_FORMAT,
      },
      multisample: {
        count: 1,
      }
    });
  }

  onResize(width, height) {
    if (!this.device) return;

    const msaaColorTexture = this.device.createTexture({
      size: { width, height },
      sampleCount: SAMPLE_COUNT,
      format: this.contextFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.colorAttachment.view = msaaColorTexture.createView();

    const depthTexture = this.device.createTexture({
      size: { width, height },
      sampleCount: SAMPLE_COUNT,
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    this.depthAttachment.view = depthTexture.createView();

    // On every size change we need to re-compute the cluster grid.
    this.computeClusterBounds();
  }

  async setGltf(gltf) {
    const resourcePromises = [];

    for (let bufferView of gltf.bufferViews) {
      resourcePromises.push(this.initBufferView(bufferView));
    }

    for (let image of gltf.images) {
      resourcePromises.push(this.initImage(image));
    }

    for (let sampler of gltf.samplers) {
      this.initSampler(sampler);
    }

    this.initNode(gltf.scene);

    await Promise.all(resourcePromises);

    for (let material of gltf.materials) {
      this.initMaterial(material);
    }

    for (let primitive of gltf.primitives) {
      this.initPrimitive(primitive);
    }

    this.outputRenderBundles = {};
    this.xrRenderBundles = {};
    this.primitives = gltf.primitives;
  }

  async initBufferView(bufferView) {
    let usage = 0;
    if (bufferView.usage.has('vertex')) {
      usage |= GPUBufferUsage.VERTEX;
    }
    if (bufferView.usage.has('index')) {
      usage |= GPUBufferUsage.INDEX;
    }

    if (!usage) {
      return;
    }

    // Oh FFS. Buffer copies have to be 4 byte aligned, I guess. >_<
    const alignedLength = Math.ceil(bufferView.byteLength / 4) * 4;

    const gpuBuffer = this.device.createBuffer({
      size: alignedLength,
      usage: usage | GPUBufferUsage.COPY_DST
    });
    bufferView.renderData.gpuBuffer = gpuBuffer;

    // TODO: Pretty sure this can all be handled more efficiently.
    const copyBuffer = this.device.createBuffer({
      size: alignedLength,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true
    });
    const copyBufferArray = new Uint8Array(copyBuffer.getMappedRange());

    const bufferData = await bufferView.dataView;

    const srcByteArray = new Uint8Array(bufferData.buffer, bufferData.byteOffset, bufferData.byteLength);
    copyBufferArray.set(srcByteArray);
    copyBuffer.unmap();

    const commandEncoder = this.device.createCommandEncoder({});
    commandEncoder.copyBufferToBuffer(copyBuffer, 0, gpuBuffer, 0, alignedLength);
    this.device.queue.submit([commandEncoder.finish()]);
  }

  async initImage(image) {
    const result = await this.textureLoader.fromBlob(await image.blob, {colorSpace: image.colorSpace});
    image.gpuTextureView = result.texture.createView();
  }

  initSampler(sampler) {
    sampler.renderData.gpuSampler = this.device.createSampler(sampler.gpuSamplerDescriptor);
  }

  initMaterial(material) {
    vec4.copy(baseColorFactor, material.baseColorFactor);
    vec2.copy(metallicRoughnessFactor, material.metallicRoughnessFactor);
    vec3.copy(emissiveFactor, material.emissiveFactor);

    const materialBuffer = this.device.createBuffer({
      size: materialUniforms.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(materialBuffer, 0, materialUniforms);

    const materialBindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayouts.material,
      entries: [{
        binding: 0,
        resource: {
          buffer: materialBuffer,
        },
      },
      {
        binding: 1,
        // TODO: Do we really need to pass one sampler per texture for accuracy? :(
        resource: material.baseColorTexture.sampler.renderData.gpuSampler,
      },
      {
        binding: 2,
        resource: material.baseColorTexture ? material.baseColorTexture.image.gpuTextureView : this.whiteTextureView,
      },
      {
        binding: 3,
        resource: material.normalTexture ? material.normalTexture.image.gpuTextureView : this.blueTextureView,
      },
      {
        binding: 4,
        resource: material.metallicRoughnessTexture ? material.metallicRoughnessTexture.image.gpuTextureView : this.whiteTextureView,
      },
      {
        binding: 5,
        resource: material.occlusionTexture ? material.occlusionTexture.image.gpuTextureView : this.whiteTextureView,
      },
      {
        binding: 6,
        resource: material.emissiveTexture ? material.emissiveTexture.image.gpuTextureView : this.blackTextureView,
      }],
    });

    material.renderData.gpuBindGroup = materialBindGroup;
  }

  initPrimitive(primitive) {
    const bufferSize = 16 * 4;

    // TODO: Support multiple instances
    if (primitive.renderData.instances.length) {
      const modelBuffer = this.device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this.device.queue.writeBuffer(modelBuffer, 0, primitive.renderData.instances[0]);

      const modelBindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayouts.primitive,
        entries: [{
          binding: 0,
          resource: {
            buffer: modelBuffer,
          },
        }],
      });

      primitive.renderData.gpuBindGroup = modelBindGroup;
    }
  }

  initNode(node) {
    for (let primitive of node.primitives) {
      if (!primitive.renderData.instances) {
        primitive.renderData.instances = [];
      }
      primitive.renderData.instances.push(node.worldMatrix);
    }

    for (let childNode of node.children) {
      this.initNode(childNode);
    }
  }

  computeClusterBounds() {
    if (!this.clusterPipeline) {
      const clusterStorageBindGroupLayout = this.device.createBindGroupLayout({
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        }]
      });

      this.clusterPipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [
            this.bindGroupLayouts.frame, // set 0
            clusterStorageBindGroupLayout, // set 1
          ]
        }),
        compute: {
          module: this.device.createShaderModule({ code: ClusterBoundsSource, label: "Cluster Bounds" }),
          entryPoint: 'main',
        }
      });

      this.clusterBuffer = this.device.createBuffer({
        size: TOTAL_TILES * 32, // Cluster x, y, z size * 32 bytes per cluster.
        usage: GPUBufferUsage.STORAGE
      });

      this.clusterStorageBindGroup = this.device.createBindGroup({
        layout: clusterStorageBindGroupLayout,
        entries: [{
          binding: 0,
          resource: {
            buffer: this.clusterBuffer,
          },
        }],
      });

      this.bindGroups.cluster = this.device.createBindGroup({
        layout: this.bindGroupLayouts.cluster,
        entries: [{
          binding: 0,
          resource: {
            buffer: this.clusterBuffer,
          },
        }],
      });
    }

    // Update the Projection uniforms. These only need to be updated on resize.
    this.device.queue.writeBuffer(this.projectionBuffer, 0, this.frameUniforms.buffer, 0, ProjectionUniformsSize);

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.clusterPipeline);
    passEncoder.setBindGroup(BIND_GROUP.Frame, this.bindGroups.frame);
    passEncoder.setBindGroup(1, this.clusterStorageBindGroup);
    passEncoder.dispatchWorkgroups(...DISPATCH_SIZE);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  computeClusterLights(commandEncoder) {
    // On every size change we need to re-compute the cluster grid.
    if (!this.clusterLightsPipeline) {
      const clusterLightsPipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [
          this.bindGroupLayouts.frame, // set 0
          this.bindGroupLayouts.cluster, // set 1
        ]
      });

      this.clusterLightsPipeline = this.device.createComputePipeline({
        layout: clusterLightsPipelineLayout,
        compute: {
          module: this.device.createShaderModule({ code: ClusterLightsSource, label: "Cluster Lights" }),
          entryPoint: 'main',
        }
      });
    }

    // Reset the light offset counter to 0 before populating the light clusters.
    this.device.queue.writeBuffer(this.clusterLightsBuffer, 0, emptyArray);

    // Update the FrameUniforms buffer with the values that are used by every
    // program and don't change for the duration of the frame.
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.clusterLightsPipeline);
    passEncoder.setBindGroup(BIND_GROUP.Frame, this.bindGroups.frame);
    passEncoder.setBindGroup(1, this.bindGroups.cluster);
    passEncoder.dispatchWorkgroups(...DISPATCH_SIZE);
    passEncoder.end();
  }

  onFrame(timestamp) {
    // TODO: If we want multisampling this should attach to the resolveTarget,
    // but there seems to be a bug with that right now?
    this.colorAttachment.resolveTarget = this.context.getCurrentTexture().createView();

    // Update the View uniforms buffer with the values. These are used by most shader programs
    // and don't change for the duration of the frame.
    this.device.queue.writeBuffer(this.viewBuffer, 0, this.frameUniforms.buffer, ProjectionUniformsSize, ViewUniformsSize);

    // Update the light unform buffer with the latest values as well.
    this.device.queue.writeBuffer(this.lightsBuffer, 0, this.lightManager.uniformArray);

    // Create a render bundle for the requested output type if one doesn't already exist.
    let renderBundle = this.outputRenderBundles[this.outputType];
    if (!renderBundle && this.primitives) {
      const helperConstructor = this.outputHelpers[this.outputType];
      const renderBundleHelper = new helperConstructor(this);
      renderBundle = this.outputRenderBundles[this.outputType] = renderBundleHelper.createRenderBundle(this.primitives);
    }

    const commandEncoder = this.device.createCommandEncoder({});

    switch (this.outputType) {
      case "lights-per-cluster":
      case "clustered-forward":
        this.computeClusterLights(commandEncoder);
        break;
    }

    const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

    if (renderBundle) {
      passEncoder.executeBundles([renderBundle]);
    }

    if (this.lightManager.render) {
      // Last, render a sprite for all of the lights. This is done using instancing so it's a single
      // call for every light.
      passEncoder.setPipeline(this.lightSpritePipeline);
      passEncoder.setBindGroup(BIND_GROUP.Frame, this.bindGroups.frame);
      passEncoder.draw(4, this.lightManager.lightCount, 0, 0);
    }

    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  async checkXRSupport(gui) {
    const vrFolder = gui.addFolder('WebXR');
    vrFolder.add({ enterVR: () => this.startXRSession() }, 'enterVR').name('Enter VR');

    if (!('xr' in navigator)) {
      console.warn('WebXR: navigator.xr not available');
    } else if (!('XRGPUBinding' in window)) {
      console.warn('WebXR: XRGPUBinding not available (WebGPU+WebXR integration not supported)');
    } else {
      try {
        const vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
        console.log('WebXR: immersive-vr supported =', vrSupported);
      } catch (err) {
        console.error('WebXR: Error checking session support:', err);
      }
    }
  }

  async startXRSession() {
    if (this.xrSession) {
      this.xrSession.end();
      return;
    }

    try {
      this.xrSession = await navigator.xr.requestSession('immersive-vr', {
        requiredFeatures: ['webgpu', 'local-floor'],
        optionalFeatures: ['transient-pointer'],
      });

      this.xrBinding = new XRGPUBinding(this.xrSession, this.device);

      this.xrLayer = this.xrBinding.createProjectionLayer({
        colorFormat: this.contextFormat,
        scaleFactor: 1.0,
      });

      this.xrSession.updateRenderState({ layers: [this.xrLayer] });

      this.xrBaseRefSpace = await this.xrSession.requestReferenceSpace('local-floor');
      // Apply player offset (for teleportation)
      this.updateXRRefSpace();

      this.xrSession.addEventListener('end', () => this.onXRSessionEnd());
      this.xrPrevTime = performance.now();
      this.xrSession.requestAnimationFrame((time, frame) => this.xrFrameCallback(time, frame));

      console.log('XR session started');
    } catch (err) {
      console.error('Failed to start XR session:', err);
    }
  }

  onXRSessionEnd() {
    this.xrSession = null;
    this.xrBinding = null;
    this.xrLayer = null;
    this.xrRefSpace = null;
    this.xrBaseRefSpace = null;
    for (const tex of this.xrDepthTextures) tex.destroy();
    this.xrDepthTextures = [];
    this.selectedLightIndex = -1;
    this.releasingLightIndex = -1;
    this.teleportTarget = null;
    this.isTeleporting = false;
    this.rafId = requestAnimationFrame(this.frameCallback);
  }

  updateXRRefSpace() {
    if (!this.xrBaseRefSpace) return;
    const offset = new XRRigidTransform({
      x: -this.playerOffset[0],
      y: -this.playerOffset[1] - 1.0, // -1.0 to spawn at reasonable height
      z: -this.playerOffset[2]
    });
    this.xrRefSpace = this.xrBaseRefSpace.getOffsetReferenceSpace(offset);
  }

  getXRDepthTexture(index, width, height) {
    const existing = this.xrDepthTextures[index];
    if (existing && existing.width === width && existing.height === height) {
      return existing;
    }
    if (existing) existing.destroy();
    this.xrDepthTextures[index] = this.device.createTexture({
      size: [width, height],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return this.xrDepthTextures[index];
  }

  xrFrameCallback(time, frame) {
    if (!this.xrSession || !this.xrRefSpace || !this.xrBinding || !this.xrLayer) return;

    this.xrSession.requestAnimationFrame((t, f) => this.xrFrameCallback(t, f));

    const pose = frame.getViewerPose(this.xrRefSpace);
    if (!pose) return;

    const timeDelta = time - this.xrPrevTime;
    this.xrPrevTime = time;

    // Handle transient-pointer for light selection (using gaze ray)
    this.handleTransientPointer(frame, pose);

    // Update wandering lights (skip selected light)
    switch (this.lightPattern) {
      case 'wandering':
        this.updateWanderingLights(timeDelta, this.selectedLightIndex);
        break;
    }

    // Update light buffer
    this.device.queue.writeBuffer(this.lightsBuffer, 0, this.lightManager.uniformArray);

    // Write per-eye uniforms BEFORE render passes
    const subImages = [];
    for (let i = 0; i < pose.views.length; i++) {
      const view = pose.views[i];
      subImages[i] = this.xrBinding.getViewSubImage(this.xrLayer, view);

      // Projection uniforms for this eye
      const proj = view.projectionMatrix;
      mat4.invert(this.xrInverseProjection, proj);
      this.xrProjectionData.set(proj, 0);
      this.xrProjectionData.set(this.xrInverseProjection, 16);
      // outputSize
      const vp = subImages[i].viewport;
      this.xrProjectionData[32] = vp.width;
      this.xrProjectionData[33] = vp.height;
      // zNear, zFar (extract from projection matrix)
      this.xrProjectionData[34] = this.zRange[0];
      this.xrProjectionData[35] = this.zRange[1];
      this.device.queue.writeBuffer(this.xrProjectionBuffers[i], 0, this.xrProjectionData);

      // View matrix and camera position for this eye
      const viewMatrix = view.transform.inverse.matrix;
      this.xrViewData.set(viewMatrix, 0);
      this.xrViewData[16] = view.transform.position.x;
      this.xrViewData[17] = view.transform.position.y;
      this.xrViewData[18] = view.transform.position.z;
      this.device.queue.writeBuffer(this.xrViewBuffers[i], 0, this.xrViewData);

      // Reticle uniforms for this eye (write BEFORE command encoder)
      if (this.teleportTarget) {
        const viewProjMatrix = mat4.multiply(mat4.create(), view.projectionMatrix, view.transform.inverse.matrix);
        this.reticleUniformData.set(viewProjMatrix, 0);
        this.reticleUniformData[16] = this.teleportTarget[0];
        this.reticleUniformData[17] = this.teleportTarget[1];
        this.reticleUniformData[18] = this.teleportTarget[2];
        this.reticleUniformData[19] = 0.5; // radius
        this.device.queue.writeBuffer(this.reticleUniformBuffers[i], 0, this.reticleUniformData);
      }
    }

    const commandEncoder = this.device.createCommandEncoder();

    // Render each eye (using naive-forward, no cluster computation needed)
    for (let i = 0; i < pose.views.length; i++) {
      const subImage = subImages[i];

      const colorView = subImage.colorTexture.createView(
        subImage.getViewDescriptor?.() ?? {}
      );
      const depthTex = this.getXRDepthTexture(i, subImage.colorTexture.width, subImage.colorTexture.height);

      const passEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: colorView,
          clearValue: { r: 0.0, g: 0.0, b: 0.5, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depthTex.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        }
      });

      const vp = subImage.viewport;
      passEncoder.setViewport(vp.x, vp.y, vp.width, vp.height, 0.0, 1.0);

      // Render scene with per-eye bind group
      this.renderXREye(passEncoder, i);

      passEncoder.end();
    }

    this.device.queue.submit([commandEncoder.finish()]);
  }

  renderXREye(passEncoder, eyeIndex) {
    // Force naive-forward in XR (clustered has view-space mismatch issues)
    const xrOutputType = 'naive-forward';

    // Create per-eye render bundles if needed
    if (!this.xrRenderBundles) {
      this.xrRenderBundles = {};
    }
    const bundleKey = `${xrOutputType}_eye${eyeIndex}`;
    let renderBundle = this.xrRenderBundles[bundleKey];

    if (!renderBundle && this.primitives) {
      const helperConstructor = this.outputHelpers[xrOutputType];
      // Temporarily override settings for XR bundle creation
      const originalFrameBindGroup = this.bindGroups.frame;
      const originalDescriptor = this.renderBundleDescriptor;
      this.bindGroups.frame = this.xrFrameBindGroups[eyeIndex];
      this.renderBundleDescriptor = this.xrRenderBundleDescriptor;

      const renderBundleHelper = new helperConstructor(this);
      renderBundle = this.xrRenderBundles[bundleKey] = renderBundleHelper.createRenderBundle(this.primitives);

      this.bindGroups.frame = originalFrameBindGroup;
      this.renderBundleDescriptor = originalDescriptor;
    }

    if (renderBundle) {
      passEncoder.executeBundles([renderBundle]);
    }

    if (this.lightManager.render) {
      passEncoder.setPipeline(this.xrLightSpritePipeline);
      passEncoder.setBindGroup(BIND_GROUP.Frame, this.xrFrameBindGroups[eyeIndex]);
      passEncoder.draw(4, this.lightManager.lightCount, 0, 0);
    }

    // Draw teleport reticle if teleporting (uniforms already written per-eye)
    if (this.teleportTarget) {
      passEncoder.setPipeline(this.reticlePipeline);
      passEncoder.setBindGroup(0, this.reticleBindGroups[eyeIndex]);
      passEncoder.draw(4);
    }
  }

  handleTransientPointer(frame, viewerPose) {
    if (!this.xrSession || !viewerPose) return;

    // Lerp releasing light color back to original
    if (this.releasingLightIndex >= 0) {
      const light = this.lightManager.lights[this.releasingLightIndex];
      vec3.lerp(light.color, light.color, this.releasingLightColor, 0.08);
      const diff = Math.abs(light.color[0] - this.releasingLightColor[0]) +
                   Math.abs(light.color[1] - this.releasingLightColor[1]) +
                   Math.abs(light.color[2] - this.releasingLightColor[2]);
      if (diff < 0.01) {
        vec3.copy(light.color, this.releasingLightColor);
        this.releasingLightIndex = -1;
      }
    }

    // Collect all transient pointers with their rays
    const transientPointers = [];
    for (const input of this.xrSession.inputSources) {
      if (input.targetRayMode === 'transient-pointer') {
        const pose = frame.getPose(input.targetRaySpace, this.xrRefSpace);
        if (pose) {
          // Extract ray from transient pointer pose (this IS the gaze ray on Vision Pro)
          const origin = [
            pose.transform.position.x,
            pose.transform.position.y,
            pose.transform.position.z
          ];
          // Get ray direction from orientation quaternion
          const q = pose.transform.orientation;
          const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
          const ix = -qy, iy = qx, iz = -qw, iw = qz;
          const dir = [
            ix * qw + iw * -qx + iy * -qz - iz * -qy,
            iy * qw + iw * -qy + iz * -qx - ix * -qz,
            iz * qw + iw * -qz + ix * -qy - iy * -qx
          ];
          transientPointers.push({ input, pose, origin, dir });
        }
      }
    }

    const primaryActive = transientPointers.length > 0;
    const secondaryActive = transientPointers.length > 1;

    // Use transient pointer ray for targeting (NOT viewer pose)
    let gazeOrigin = null;
    let gazeDir = null;
    if (primaryActive) {
      gazeOrigin = transientPointers[0].origin;
      gazeDir = transientPointers[0].dir;
    }

    // Test gaze ray against lights (only if we have a transient pointer)
    let closestLightDist = Infinity;
    let closestLightIdx = -1;
    let groundHitT = Infinity;
    let groundHitPos = null;

    if (gazeOrigin && gazeDir) {
      // Proper ray-sphere intersection (quadratic formula)
      const hitRadius = 0.5; // Visual sprite hit radius (not light.range!)
      
      for (let i = 4; i < this.lightManager.lightCount; i++) {
        const light = this.lightManager.lights[i];
        // offset = origin - center
        const ox = gazeOrigin[0] - light.position[0];
        const oy = gazeOrigin[1] - light.position[1];
        const oz = gazeOrigin[2] - light.position[2];
        
        // a = ray . ray (should be 1 if normalized, but compute anyway)
        const a = gazeDir[0] * gazeDir[0] + gazeDir[1] * gazeDir[1] + gazeDir[2] * gazeDir[2];
        // b = 2 * ray . offset
        const b = 2 * (gazeDir[0] * ox + gazeDir[1] * oy + gazeDir[2] * oz);
        // c = offset . offset - radius^2
        const c = ox * ox + oy * oy + oz * oz - hitRadius * hitRadius;
        
        const discriminant = b * b - 4 * a * c;
        if (discriminant > 0) {
          const t = (-b - Math.sqrt(discriminant)) / (2 * a);
          if (t > 0 && t < closestLightDist) {
            closestLightDist = t;
            closestLightIdx = i;
          }
        }
      }

      // Test gaze ray against ground
      if (gazeDir[1] < -0.01) {
        const t = -gazeOrigin[1] / gazeDir[1];
        if (t > 0 && t < 100) {
          groundHitT = t;
          groundHitPos = [gazeOrigin[0] + gazeDir[0] * t, 0, gazeOrigin[2] + gazeDir[2] * t];
        }
      }
    }

    if (primaryActive) {
      const rayOrigin = transientPointers[0].origin;
      const rayDir = transientPointers[0].dir;
      const handPos = transientPointers[0].pose.transform.position;

      if (!this.wasTransientPointerActive) {
        // Pinch start
        if (closestLightIdx >= 0 && closestLightDist < groundHitT) {
          // Light mode
          this.isTeleporting = false;
          if (this.releasingLightIndex === closestLightIdx) this.releasingLightIndex = -1;
          this.selectedLightIndex = closestLightIdx;
          const light = this.lightManager.lights[closestLightIdx];
          vec3.copy(this.originalLightColor, light.color);
          this.selectionStartLightY = light.position[1];
          this.selectionStartRayDirY = rayDir[1];
          // Capture XZ distance from ray origin to light
          const dx = light.position[0] - rayOrigin[0];
          const dz = light.position[2] - rayOrigin[2];
          this.selectionStartDistance = Math.sqrt(dx * dx + dz * dz);
          this.colorAmplify = 1.0;
          this.secondHandActive = false;
        } else if (groundHitPos) {
          // Teleport mode
          this.isTeleporting = true;
          this.teleportTarget = [...groundHitPos];
          this.teleportStartPos = [...groundHitPos];
          this.teleportStartRayOrigin = [...rayOrigin];
        }
      } else if (this.isTeleporting && this.teleportTarget) {
        // Teleport: adjust target based on ray origin movement
        const multiplier = 10.0;
        this.teleportTarget[0] = this.teleportStartPos[0] + (rayOrigin[0] - this.teleportStartRayOrigin[0]) * multiplier;
        this.teleportTarget[2] = this.teleportStartPos[2] + (rayOrigin[2] - this.teleportStartRayOrigin[2]) * multiplier;
      } else if (!this.isTeleporting && this.selectedLightIndex >= 0) {
        // Light movement (webgpu-water style)
        const light = this.lightManager.lights[this.selectedLightIndex];

        // XZ: horizontal ray direction at fixed distance from ray origin
        const flatDirX = rayDir[0];
        const flatDirZ = rayDir[2];
        const flatLen = Math.sqrt(flatDirX * flatDirX + flatDirZ * flatDirZ);
        if (flatLen > 0.001) {
          const normX = flatDirX / flatLen;
          const normZ = flatDirZ / flatLen;
          light.position[0] = rayOrigin[0] + normX * this.selectionStartDistance;
          light.position[2] = rayOrigin[2] + normZ * this.selectionStartDistance;
        }

        // Y: arm pivot (ray direction Y delta)
        const rayDirYDelta = (rayDir[1] - this.selectionStartRayDirY) * 5.0;
        light.position[1] = Math.max(0.2, this.selectionStartLightY + rayDirYDelta);

        // Amplify color while selected
        this.colorAmplify = Math.min(this.colorAmplify + 0.05, 2.0);
        vec3.scale(light.color, this.originalLightColor, this.colorAmplify);

        // Two-handed color control
        if (secondaryActive) {
          const secondHand = transientPointers[1].pose.transform.position;
          if (!this.secondHandActive) {
            this.secondHandActive = true;
            this.secondHandStartX = secondHand.x;
          }
          // Map hand X movement to color ramp
          const delta = (secondHand.x - this.secondHandStartX) * 5.0;
          const t = Math.max(0, Math.min(1, (delta + 1) / 2)); // -1 to 1 -> 0 to 1
          const rampIdx = t * (this.colorRamp.length - 1);
          const idx0 = Math.floor(rampIdx);
          const idx1 = Math.min(idx0 + 1, this.colorRamp.length - 1);
          const blend = rampIdx - idx0;
          vec3.lerp(this.originalLightColor, this.colorRamp[idx0], this.colorRamp[idx1], blend);
          vec3.scale(light.color, this.originalLightColor, this.colorAmplify);
        } else {
          this.secondHandActive = false;
        }
      }
    } else if (this.wasTransientPointerActive) {
      // Release
      if (this.isTeleporting && this.teleportTarget) {
        // SET the offset to teleport target (not add!)
        this.playerOffset[0] = this.teleportTarget[0];
        this.playerOffset[2] = this.teleportTarget[2];
        this.updateXRRefSpace();
        this.teleportTarget = null;
        this.teleportStartPos = null;
        this.teleportStartRayOrigin = null;
        this.isTeleporting = false;
      } else if (this.selectedLightIndex >= 0) {
        this.releasingLightIndex = this.selectedLightIndex;
        vec3.copy(this.releasingLightColor, this.originalLightColor);
        this.selectedLightIndex = -1;
        this.secondHandActive = false;
      }
    }

    this.wasTransientPointerActive = primaryActive;
  }

  getTeleportTarget() {
    return this.teleportTarget;
  }
}