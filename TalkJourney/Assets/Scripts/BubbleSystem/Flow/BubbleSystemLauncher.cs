using TalkJourney.BubbleSystem.Data;
using UnityEngine;

namespace TalkJourney.BubbleSystem.Flow
{
    /// <summary>
    /// Creates/uses a BubbleSystemBootstrap instance on demand and starts a selected stage.
    /// Wire stage buttons to StartStageFromButton with different StageData assets.
    /// </summary>
    [DisallowMultipleComponent]
    public class BubbleSystemLauncher : MonoBehaviour
    {
        [System.Serializable]
        public class StageLaunchBinding
        {
            public StageData stageData;
            public Transform stageRuntimeParent;
        }

        [Header("Bootstrap Source")]
        [Tooltip("Optional existing bootstrap already in scene. If null, bootstrapPrefab is instantiated on first start.")]
        public BubbleSystemBootstrap existingBootstrap;

        [Tooltip("Prefab instantiated when no existing bootstrap is assigned.")]
        public BubbleSystemBootstrap bootstrapPrefab;

        [Tooltip("Optional default parent for instantiated bootstrap when no stageRuntimeParent is passed.")]
        public Transform runtimeParent;

        [Header("Button Bindings")]
        [Tooltip("Configure stage + canvas pairs here, then call StartStageFromBindingIndex(index) from a UI Button.")]
        public StageLaunchBinding[] stageLaunchBindings;

        [Header("Results Presenter")]
        [Tooltip("Optional existing JourneyResultsPresenter already in scene.")]
        public JourneyResultsPresenter existingResultsPresenter;

        [Tooltip("Optional presenter prefab instantiated when existingResultsPresenter is not assigned.")]
        public JourneyResultsPresenter resultsPresenterPrefab;

        private BubbleSystemBootstrap _activeBootstrap;
        private JourneyResultsPresenter _activeResultsPresenter;

        private void Awake()
        {
            _activeBootstrap = existingBootstrap;
            _activeResultsPresenter = existingResultsPresenter;
        }

        public void StartStageFromButton(StageData stageData, Transform stageRuntimeParent)
        {
            if (stageData == null)
            {
                Debug.LogWarning("BubbleSystemLauncher.StartStageFromButton called with null StageData.", this);
                return;
            }

            var resolvedParent = stageRuntimeParent != null ? stageRuntimeParent : runtimeParent;

            if (_activeBootstrap != null && _activeBootstrap != existingBootstrap && _activeBootstrap.transform.parent != resolvedParent)
            {
                StopAndDestroyActiveBootstrap();
            }

            var bootstrap = ResolveBootstrap(resolvedParent);
            if (bootstrap == null)
            {
                Debug.LogError("BubbleSystemLauncher could not resolve BubbleSystemBootstrap. Assign existingBootstrap or bootstrapPrefab.", this);
                return;
            }

            EnsureResultsPresenter(resolvedParent);

            bootstrap.StartStage(stageData);
        }

        /// <summary>
        /// Unity UI Button friendly overload: pass one integer index from OnClick.
        /// stageData and stageRuntimeParent are read from stageLaunchBindings[index].
        /// </summary>
        public void StartStageFromBindingIndex(int bindingIndex)
        {
            if (stageLaunchBindings == null || stageLaunchBindings.Length == 0)
            {
                Debug.LogWarning("BubbleSystemLauncher has no stageLaunchBindings configured.", this);
                return;
            }

            if (bindingIndex < 0 || bindingIndex >= stageLaunchBindings.Length)
            {
                Debug.LogWarning($"BubbleSystemLauncher binding index out of range: {bindingIndex}.", this);
                return;
            }

            var binding = stageLaunchBindings[bindingIndex];
            if (binding == null)
            {
                Debug.LogWarning($"BubbleSystemLauncher binding at index {bindingIndex} is null.", this);
                return;
            }

            StartStageFromButton(binding.stageData, binding.stageRuntimeParent);
        }

        public void StartInitialStageFromButton()
        {
            var resolvedParent = runtimeParent;

            if (_activeBootstrap != null && _activeBootstrap != existingBootstrap && _activeBootstrap.transform.parent != resolvedParent)
            {
                StopAndDestroyActiveBootstrap();
            }

            var bootstrap = ResolveBootstrap(resolvedParent);
            if (bootstrap == null)
            {
                Debug.LogError("BubbleSystemLauncher could not resolve BubbleSystemBootstrap. Assign existingBootstrap or bootstrapPrefab.", this);
                return;
            }

            EnsureResultsPresenter(resolvedParent);

            bootstrap.StartInitialStage();
        }

        public void StopAndDestroyActiveBootstrap()
        {
            if (_activeBootstrap == null)
            {
                return;
            }

            if (_activeBootstrap == existingBootstrap)
            {
                return;
            }

            Destroy(_activeBootstrap.gameObject);
            _activeBootstrap = null;

            if (_activeResultsPresenter != null && _activeResultsPresenter != existingResultsPresenter)
            {
                Destroy(_activeResultsPresenter.gameObject);
                _activeResultsPresenter = null;
            }
        }

        private BubbleSystemBootstrap ResolveBootstrap(Transform spawnParent)
        {
            if (_activeBootstrap != null)
            {
                return _activeBootstrap;
            }

            if (existingBootstrap != null)
            {
                _activeBootstrap = existingBootstrap;
                return _activeBootstrap;
            }

            if (bootstrapPrefab == null)
            {
                return null;
            }

            _activeBootstrap = spawnParent != null
                ? Instantiate(bootstrapPrefab, spawnParent)
                : Instantiate(bootstrapPrefab);

            _activeBootstrap.initializeOnAwake = false;
            _activeBootstrap.EnsureSetup();
            return _activeBootstrap;
        }

        private JourneyResultsPresenter EnsureResultsPresenter(Transform spawnParent)
        {
            if (_activeResultsPresenter != null)
            {
                return _activeResultsPresenter;
            }

            if (existingResultsPresenter != null)
            {
                _activeResultsPresenter = existingResultsPresenter;
                return _activeResultsPresenter;
            }

            if (resultsPresenterPrefab == null)
            {
                return null;
            }

            _activeResultsPresenter = spawnParent != null
                ? Instantiate(resultsPresenterPrefab, spawnParent)
                : Instantiate(resultsPresenterPrefab);

            return _activeResultsPresenter;
        }

    }
}
