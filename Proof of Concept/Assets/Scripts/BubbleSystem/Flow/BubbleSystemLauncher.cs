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
        [Header("Bootstrap Source")]
        [Tooltip("Optional existing bootstrap already in scene. If null, bootstrapPrefab is instantiated on first start.")]
        public BubbleSystemBootstrap existingBootstrap;

        [Tooltip("Prefab instantiated when no existing bootstrap is assigned.")]
        public BubbleSystemBootstrap bootstrapPrefab;

        [Tooltip("Optional parent for instantiated bootstrap.")]
        public Transform runtimeParent;

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

        public void StartStageFromButton(StageData stageData)
        {
            if (stageData == null)
            {
                Debug.LogWarning("BubbleSystemLauncher.StartStageFromButton called with null StageData.", this);
                return;
            }

            var bootstrap = ResolveBootstrap();
            if (bootstrap == null)
            {
                Debug.LogError("BubbleSystemLauncher could not resolve BubbleSystemBootstrap. Assign existingBootstrap or bootstrapPrefab.", this);
                return;
            }

            EnsureResultsPresenter();

            bootstrap.StartStage(stageData);
        }

        public void StartInitialStageFromButton()
        {
            var bootstrap = ResolveBootstrap();
            if (bootstrap == null)
            {
                Debug.LogError("BubbleSystemLauncher could not resolve BubbleSystemBootstrap. Assign existingBootstrap or bootstrapPrefab.", this);
                return;
            }

            EnsureResultsPresenter();

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

        private BubbleSystemBootstrap ResolveBootstrap()
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

            _activeBootstrap = runtimeParent != null
                ? Instantiate(bootstrapPrefab, runtimeParent)
                : Instantiate(bootstrapPrefab);

            _activeBootstrap.initializeOnAwake = false;
            _activeBootstrap.EnsureSetup();
            return _activeBootstrap;
        }

        private JourneyResultsPresenter EnsureResultsPresenter()
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

            _activeResultsPresenter = runtimeParent != null
                ? Instantiate(resultsPresenterPrefab, runtimeParent)
                : Instantiate(resultsPresenterPrefab);

            return _activeResultsPresenter;
        }
    }
}
