using UnityEngine;
using System.Collections.Generic;

namespace ZeukGames { 

[AddComponentMenu("Zeuk Games/Dynamo Bone")]

    public class DynamoBone : MonoBehaviour
    {
        public Transform Root = null;
        public float UpdateRate = 60.0f;

        [Range(0, 1)] public float Damping = 0.2f;
        [Range(0, 1)] public float Elasticity = 0.05f;
        [Range(0, 1)] public float Stiffness = 0.7f;
        [Range(0, 1)] public float Inertia = 0.5f;

        public float EndLength = 0;
        public Vector3 EndOffset = Vector3.zero;
        public Vector3 Gravity = Vector3.zero;
        public Vector3 ExternalForce = Vector3.zero;

        public enum AxisConstraint { None, X, Y, Z }
        public AxisConstraint FreezeAxis = AxisConstraint.None;

        public bool EnableDistanceCulling = false;
        public Transform DistanceReference = null;
        public float CullingDistance = 20;

        private Vector3 localGravity = Vector3.zero;
        private Vector3 objectMovement = Vector3.zero;
        private Vector3 previousPosition = Vector3.zero;
        private float objectScale = 1.0f;
        private float updateTimer = 0;
        private float blendWeight = 1.0f;
        private bool isCulled = false;

        private class BoneParticle
        {
            public Transform Bone = null;
            public int ParentIndex = -1;
            public float Damping = 0;
            public float Elasticity = 0;
            public float Stiffness = 0;
            public float Inertia = 0;
            public float Length = 0;

            public Vector3 Position = Vector3.zero;
            public Vector3 PreviousPosition = Vector3.zero;
            public Vector3 EndOffset = Vector3.zero;
            public Vector3 InitialLocalPosition = Vector3.zero;
            public Quaternion InitialLocalRotation = Quaternion.identity;
        }

        private readonly List<BoneParticle> particles = new List<BoneParticle>();

        void Start() => InitializeParticles();
        void Update() => TryInitializeTransforms();
        void LateUpdate() => UpdateSimulation();

        #region Public API
        public void SetBlendWeight(float weight)
        {
            if (blendWeight != weight)
            {
                if (weight == 0)
                    RestoreTransforms();
                else if (blendWeight == 0)
                    ResetParticlePositions();

                blendWeight = weight;
            }
        }

        public float GetBlendWeight() => blendWeight;
        #endregion

        #region MonoBehaviour Helpers
        private void TryInitializeTransforms()
        {
            if (blendWeight > 0 && !(EnableDistanceCulling && isCulled))
                RestoreTransforms();
        }

        private void UpdateSimulation()
        {
            if (EnableDistanceCulling)
                EvaluateCulling();

            if (blendWeight > 0 && !(EnableDistanceCulling && isCulled))
                SimulateDynamics(Time.deltaTime);
        }
        #endregion

        #region Culling
        private void EvaluateCulling()
        {
            Transform reference = DistanceReference ?? Camera.main?.transform;
            if (reference == null) return;

            float distanceSqr = (reference.position - transform.position).sqrMagnitude;
            bool shouldCull = distanceSqr > CullingDistance * CullingDistance;

            if (shouldCull != isCulled)
            {
                if (!shouldCull)
                    ResetParticlePositions();
                isCulled = shouldCull;
            }
        }
        #endregion

        void OnEnable() => ResetParticlePositions();
        void OnDisable() => RestoreTransforms();

        void OnValidate()
        {
            UpdateRate = Mathf.Max(UpdateRate, 0);
            Damping = Mathf.Clamp01(Damping);
            Elasticity = Mathf.Clamp01(Elasticity);
            Stiffness = Mathf.Clamp01(Stiffness);
            Inertia = Mathf.Clamp01(Inertia);

#if UNITY_EDITOR
            if (Application.isEditor && Application.isPlaying)
            {
                RestoreTransforms();
                InitializeParticles();
            }
#endif
        }

        void OnDrawGizmosSelected()
        {
            if (!enabled || Root == null) return;

#if UNITY_EDITOR
            if (!Application.isPlaying && transform.hasChanged)
            {
                RestoreTransforms();
                InitializeParticles();
            }
#endif

            Gizmos.color = Color.white;
            foreach (var p in particles)
            {
                if (p.ParentIndex >= 0)
                {
                    var parent = particles[p.ParentIndex];
                    Gizmos.DrawLine(p.Position, parent.Position);
                }
            }
        }

        #region Initialization
        private void InitializeParticles()
        {
            particles.Clear();
            if (Root == null) return;

            localGravity = Root.InverseTransformDirection(Gravity);
            objectScale = transform.lossyScale.x;
            previousPosition = transform.position;
            objectMovement = Vector3.zero;

            AppendBoneChain(Root, -1, 0);
            UpdateParticleProperties();
        }

        private void AppendBoneChain(Transform current, int parentIndex, float currentLength)
        {
            var particle = new BoneParticle
            {
                Bone = current,
                ParentIndex = parentIndex,
                Position = current != null ? current.position : Vector3.zero,
                PreviousPosition = current != null ? current.position : Vector3.zero,
                InitialLocalPosition = current != null ? current.localPosition : Vector3.zero,
                InitialLocalRotation = current != null ? current.localRotation : Quaternion.identity
            };

            if (parentIndex >= 0 && current != null)
            {
                currentLength += (particles[parentIndex].Bone.position - current.position).magnitude;
                particle.Length = currentLength;
            }
            else if (parentIndex >= 0 && current == null)
            {
                var parent = particles[parentIndex].Bone;
                particle.EndOffset = EndLength > 0
                    ? parent.InverseTransformPoint((parent.position * 2 - parent.parent.position)) * EndLength
                    : parent.InverseTransformPoint(transform.TransformDirection(EndOffset) + parent.position);
                particle.Position = particle.PreviousPosition = parent.TransformPoint(particle.EndOffset);
            }

            int index = particles.Count;
            particles.Add(particle);

            if (current != null)
            {
                for (int i = 0; i < current.childCount; i++)
                {
                    var child = current.GetChild(i);
                    AppendBoneChain(child, index, currentLength);
                }

                if (current.childCount == 0 && (EndLength > 0 || EndOffset != Vector3.zero))
                    AppendBoneChain(null, index, currentLength);
            }
        }

        private void UpdateParticleProperties()
        {
            foreach (var particle in particles)
            {
                particle.Damping = Damping;
                particle.Elasticity = Elasticity;
                particle.Stiffness = Stiffness;
                particle.Inertia = Inertia;
            }
        }
        #endregion

        #region Transform Restore & Reset
        private void RestoreTransforms()
        {
            foreach (var p in particles)
            {
                if (p.Bone == null) continue;
                p.Bone.localPosition = p.InitialLocalPosition;
                p.Bone.localRotation = p.InitialLocalRotation;
            }
        }

        private void ResetParticlePositions()
        {
            foreach (var p in particles)
            {
                if (p.Bone != null)
                    p.Position = p.PreviousPosition = p.Bone.position;
                else if (p.ParentIndex >= 0)
                    p.Position = p.PreviousPosition = particles[p.ParentIndex].Bone.TransformPoint(p.EndOffset);
            }
            previousPosition = transform.position;
        }
        #endregion

        #region Simulation
        private void SimulateDynamics(float deltaTime)
        {
            if (Root == null) return;

            objectScale = Mathf.Abs(transform.lossyScale.x);
            objectMovement = transform.position - previousPosition;
            previousPosition = transform.position;

            int loops = 1;
            if (UpdateRate > 0)
            {
                float step = 1.0f / UpdateRate;
                updateTimer += deltaTime;
                loops = 0;

                while (updateTimer >= step)
                {
                    updateTimer -= step;
                    if (++loops >= 3)
                    {
                        updateTimer = 0;
                        break;
                    }
                }
            }

            if (loops > 0)
            {
                for (int i = 0; i < loops; ++i)
                {
                    ApplyForces();
                    ConstrainParticles();
                    objectMovement = Vector3.zero;
                }
            }
            else
            {
                SkipUpdate();
            }

            ApplyTransforms();
        }

        private void ApplyForces()
        {
            Vector3 totalForce = (Gravity - Vector3.Project(Root.TransformDirection(localGravity), Gravity.normalized)) + ExternalForce;
            totalForce *= objectScale;

            foreach (var p in particles)
            {
                if (p.ParentIndex < 0)
                {
                    p.PreviousPosition = p.Position;
                    p.Position = p.Bone.position;
                    continue;
                }

                Vector3 velocity = p.Position - p.PreviousPosition;
                Vector3 inertia = objectMovement * p.Inertia;
                p.PreviousPosition = p.Position + inertia;
                p.Position += velocity * (1 - p.Damping) + totalForce + inertia;
            }
        }

        private void ConstrainParticles()
        {
            Plane constraintPlane = new Plane();

            for (int i = 1; i < particles.Count; i++)
            {
                BoneParticle p = particles[i];
                BoneParticle parent = particles[p.ParentIndex];

                float restLength = (p.Bone != null)
                    ? (parent.Bone.position - p.Bone.position).magnitude
                    : parent.Bone.localToWorldMatrix.MultiplyVector(p.EndOffset).magnitude;

                float stiffness = Mathf.Lerp(1.0f, p.Stiffness, blendWeight);
                if (stiffness > 0 || p.Elasticity > 0)
                {
                    Matrix4x4 matrix = parent.Bone.localToWorldMatrix;
                    matrix.SetColumn(3, parent.Position);
                    Vector3 restPosition = (p.Bone != null)
                        ? matrix.MultiplyPoint3x4(p.Bone.localPosition)
                        : matrix.MultiplyPoint3x4(p.EndOffset);

                    Vector3 offset = restPosition - p.Position;
                    p.Position += offset * p.Elasticity;

                    if (stiffness > 0)
                    {
                        offset = restPosition - p.Position;
                        float distance = offset.magnitude;
                        float maxDistance = restLength * (1 - stiffness) * 2;
                        if (distance > maxDistance)
                            p.Position += offset * ((distance - maxDistance) / distance);
                    }
                }

                // Freeze axis constraint
                if (FreezeAxis != AxisConstraint.None)
                {
                    Vector3 normal = Vector3.zero;
                    switch (FreezeAxis)
                    {
                        case AxisConstraint.X: normal = parent.Bone.right; break;
                        case AxisConstraint.Y: normal = parent.Bone.up; break;
                        case AxisConstraint.Z: normal = parent.Bone.forward; break;
                    }
                    constraintPlane.SetNormalAndPosition(normal, parent.Position);
                    p.Position -= constraintPlane.normal * constraintPlane.GetDistanceToPoint(p.Position);
                }

                Vector3 dir = parent.Position - p.Position;
                float len = dir.magnitude;
                if (len > 0)
                    p.Position += dir * ((len - restLength) / len);
            }
        }

        private void SkipUpdate()
        {
            for (int i = 0; i < particles.Count; i++)
            {
                BoneParticle p = particles[i];
                if (p.ParentIndex >= 0)
                {
                    p.PreviousPosition += objectMovement;
                    p.Position += objectMovement;

                    BoneParticle parent = particles[p.ParentIndex];

                    float restLength = (p.Bone != null)
                        ? (parent.Bone.position - p.Bone.position).magnitude
                        : parent.Bone.localToWorldMatrix.MultiplyVector(p.EndOffset).magnitude;

                    float stiffness = Mathf.Lerp(1.0f, p.Stiffness, blendWeight);
                    if (stiffness > 0)
                    {
                        Matrix4x4 matrix = parent.Bone.localToWorldMatrix;
                        matrix.SetColumn(3, parent.Position);
                        Vector3 restPos = (p.Bone != null)
                            ? matrix.MultiplyPoint3x4(p.Bone.localPosition)
                            : matrix.MultiplyPoint3x4(p.EndOffset);

                        Vector3 offset = restPos - p.Position;
                        float len = offset.magnitude;
                        float maxLen = restLength * (1 - stiffness) * 2;
                        if (len > maxLen)
                            p.Position += offset * ((len - maxLen) / len);
                    }

                    Vector3 delta = parent.Position - p.Position;
                    float dlen = delta.magnitude;
                    if (dlen > 0)
                        p.Position += delta * ((dlen - restLength) / dlen);
                }
                else
                {
                    p.PreviousPosition = p.Position;
                    p.Position = p.Bone.position;
                }
            }
        }

        private void ApplyTransforms()
        {
            for (int i = 1; i < particles.Count; i++)
            {
                BoneParticle p = particles[i];
                BoneParticle parent = particles[p.ParentIndex];

                if (parent.Bone.childCount <= 1)
                {
                    Vector3 dir = (p.Bone != null) ? p.Bone.localPosition : p.EndOffset;
                    Quaternion rotation = Quaternion.FromToRotation(parent.Bone.TransformDirection(dir), p.Position - parent.Position);
                    parent.Bone.rotation = rotation * parent.Bone.rotation;
                }

                if (p.Bone != null)
                    p.Bone.position = p.Position;
            }
        }
        #endregion
    }
}