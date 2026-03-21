using UnityEngine;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Bubbles;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Localization;
using TalkJourney.BubbleSystem.Speech;
using TalkJourney.GameServices;

namespace TalkJourney.BubbleSystem.Flow
{
    [DefaultExecutionOrder(-1000)]
    [DisallowMultipleComponent]
    public class BubbleSystemBootstrap : MonoBehaviour
    {
        [Header("Scene Data")]
        public StageData initialStage;
        public Transform sentenceBubbleParent;
        public Transform selectionBubbleParent;
        public DisplayBubbleController fallbackDisplayBubblePrefab;
        public SelectionBubbleController selectionBubblePrefab;
        public VrPointerInteractable speakerButtonInteractable;
        public VrPointerInteractable bypassButtonInteractable;

        [Header("External Services")]
        [Tooltip("Optional speech recognition component implementing ISpeechRecognitionService. If empty, GlobalGameServicesBootstrap.realtimeWhisper is used first.")]
        public MonoBehaviour speechRecognitionBehaviour;

        [Tooltip("Optional localization component implementing ILocalizationService. If empty, scene-wide service resolution is used.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Tooltip("Optional audio backend component implementing IAudioBackendClient. If empty, scene-wide service resolution is used.")]
        public MonoBehaviour audioBackendClientBehaviour;

        [Tooltip("Optional playback source. If empty, an AudioSource is added here.")]
        public AudioSource playbackSource;

        [Header("Service Resolution")]
        [Tooltip("Services are resolved from GlobalGameServicesBootstrap. If not found there, scene-wide fallback is used. Local service creation is disabled by design.")]
        public bool strictGlobalServiceMode = true;

        [Header("Auto Setup")]
        [Tooltip("When enabled, setup runs in Awake. Disable if this bootstrap is instantiated/started by a launcher on demand.")]
        public bool initializeOnAwake = true;

        public bool autoFindSpeechRecognitionBehaviour = true;
        public bool autoCreateAudioSource = true;

        private bool _isSetupComplete;

        private void Reset()
        {
            EnsureSetup();
        }

        private void Awake()
        {
            if (initializeOnAwake)
            {
                EnsureSetup();
            }
        }

        public void EnsureSetup()
        {
            if (_isSetupComplete)
            {
                return;
            }

            var stageController = GetOrAddComponent<StageController>();
            var sentenceBubbleController = GetOrAddComponent<SentenceBubbleController>();
            var selectionSpeechMatcher = GetOrAddComponent<SelectionSpeechMatcher>();
            var audioPlaybackManager = GetOrAddComponent<AudioPlaybackManager>();

            localizationServiceBehaviour = EnsureLocalizationService();
            audioBackendClientBehaviour = EnsureAudioBackendClient();

            if (autoCreateAudioSource && playbackSource == null)
            {
                playbackSource = GetComponent<AudioSource>();
                if (playbackSource == null)
                {
                    playbackSource = gameObject.AddComponent<AudioSource>();
                }
            }

            speechRecognitionBehaviour = EnsureSpeechRecognitionService(selectionSpeechMatcher);

            stageController.initialStage = initialStage;
            stageController.sentenceBubbleController = sentenceBubbleController;
            stageController.selectionBubbleParent = selectionBubbleParent;
            stageController.selectionBubblePrefab = selectionBubblePrefab;
            stageController.selectionSpeechMatcher = selectionSpeechMatcher;
            stageController.localizationServiceBehaviour = localizationServiceBehaviour;
            stageController.audioPlaybackManagerBehaviour = audioPlaybackManager;

            sentenceBubbleController.audioPlaybackManagerBehaviour = audioPlaybackManager;
            sentenceBubbleController.localizationServiceBehaviour = localizationServiceBehaviour;
            sentenceBubbleController.sentenceBubbleParent = sentenceBubbleParent;
            sentenceBubbleController.fallbackDisplayBubblePrefab = fallbackDisplayBubblePrefab;
            sentenceBubbleController.speakerButtonInteractable = speakerButtonInteractable;
            sentenceBubbleController.RefreshDependencies();

            selectionSpeechMatcher.speechRecognitionBehaviour = speechRecognitionBehaviour;
            selectionSpeechMatcher.bypassButtonInteractable = bypassButtonInteractable;
            selectionSpeechMatcher.RefreshDependencies();

            audioPlaybackManager.playbackSource = playbackSource;
            audioPlaybackManager.backendClientBehaviour = audioBackendClientBehaviour;
            audioPlaybackManager.RefreshDependencies();

            _isSetupComplete = true;
        }

        /// <summary>
        /// Starts BubbleSystem for a specific stage. Useful for menu buttons (Level 1, Level 2, ...).
        /// </summary>
        public void StartStage(StageData stageData)
        {
            if (stageData == null)
            {
                Debug.LogWarning("BubbleSystemBootstrap.StartStage was called with null stageData.", this);
                return;
            }

            EnsureSetup();

            var stageController = GetComponent<StageController>();
            if (stageController == null)
            {
                Debug.LogError("BubbleSystemBootstrap could not find StageController after setup.", this);
                return;
            }

            stageController.autoStartInitialStage = false;
            stageController.LoadStage(stageData);
        }

        /// <summary>
        /// Starts BubbleSystem using the configured initialStage field.
        /// </summary>
        public void StartInitialStage()
        {
            if (initialStage == null)
            {
                Debug.LogWarning("BubbleSystemBootstrap.StartInitialStage was called but initialStage is not assigned.", this);
                return;
            }

            StartStage(initialStage);
        }

        private MonoBehaviour EnsureLocalizationService()
        {
            if (localizationServiceBehaviour != null)
            {
                return localizationServiceBehaviour;
            }

            // Priority 1: Try to get from GlobalGameServicesBootstrap (persistent global services)
            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null && globalServices.localizationResolver != null)
            {
                localizationServiceBehaviour = globalServices.localizationResolver;
                return localizationServiceBehaviour;
            }

            // Priority 2: Try to find in scene if globals not available
            localizationServiceBehaviour = FindFirstSceneBehaviourImplementing<ILocalizationService>(this);
            if (localizationServiceBehaviour != null)
            {
                return localizationServiceBehaviour;
            }

            // Priority 3: Try to get from this GameObject if already added
            localizationServiceBehaviour = GetComponent<LocalizationResolver>();
            if (localizationServiceBehaviour != null)
            {
                return localizationServiceBehaviour;
            }

            // Strict mode: Fail if no service found. Local creation is disabled by design.
            Debug.LogError(
                "BubbleSystemBootstrap could not resolve ILocalizationService. " +
                "Ensure GlobalGameServicesBootstrap is set up in a persistent scene with localizationResolver assigned, or " +
                "add a LocalizationResolver to this scene before BubbleSystemBootstrap initializes.",
                this);

            return null;
        }

        private MonoBehaviour EnsureAudioBackendClient()
        {
            if (audioBackendClientBehaviour != null)
            {
                return audioBackendClientBehaviour;
            }

            // Priority 1: Try to get from GlobalGameServicesBootstrap (persistent global services)
            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null && globalServices.audioBackendClient != null)
            {
                audioBackendClientBehaviour = globalServices.audioBackendClient;
                return audioBackendClientBehaviour;
            }

            // Priority 2: Try to find in scene if globals not available
            audioBackendClientBehaviour = FindFirstSceneBehaviourImplementing<IAudioBackendClient>(this);
            if (audioBackendClientBehaviour != null)
            {
                return audioBackendClientBehaviour;
            }

            // Priority 3: Try to get from this GameObject if already added
            audioBackendClientBehaviour = GetComponent<AudioBackendClient>();
            if (audioBackendClientBehaviour != null)
            {
                return audioBackendClientBehaviour;
            }

            // Strict mode: Fail if no service found. Local creation is disabled by design.
            Debug.LogError(
                "BubbleSystemBootstrap could not resolve IAudioBackendClient. " +
                "Ensure GlobalGameServicesBootstrap is set up in a persistent scene with audioBackendClient assigned, or " +
                "add an AudioBackendClient to this scene before BubbleSystemBootstrap initializes.",
                this);

            return null;
        }

        private MonoBehaviour FindFirstSceneBehaviourImplementing<T>(Component excludedComponent) where T : class
        {
            var sceneBehaviours = FindObjectsOfType<MonoBehaviour>(true);
            for (int i = 0; i < sceneBehaviours.Length; i++)
            {
                var behaviour = sceneBehaviours[i];
                if (behaviour == null || behaviour == excludedComponent)
                {
                    continue;
                }

                if (behaviour is T)
                {
                    return behaviour;
                }
            }

            return null;
        }

        private T GetOrAddComponent<T>() where T : Component
        {
            var component = GetComponent<T>();
            if (component == null)
            {
                component = gameObject.AddComponent<T>();
            }

            return component;
        }

        private MonoBehaviour EnsureSpeechRecognitionService(Component excludedComponent)
        {
            if (speechRecognitionBehaviour != null)
            {
                return speechRecognitionBehaviour;
            }

            // Priority 1: Try to get from GlobalGameServicesBootstrap (persistent global services)
            var globalServices = GlobalGameServicesBootstrap.Instance;
            if (globalServices != null && globalServices.realtimeWhisper != null)
            {
                speechRecognitionBehaviour = globalServices.realtimeWhisper;
                return speechRecognitionBehaviour;
            }

            if (!autoFindSpeechRecognitionBehaviour)
            {
                return null;
            }

            // Priority 2: Try local/scene-wide lookup for backward compatibility
            speechRecognitionBehaviour = FindFirstBehaviourImplementing<ISpeechRecognitionService>(excludedComponent);
            if (speechRecognitionBehaviour != null)
            {
                return speechRecognitionBehaviour;
            }

            Debug.LogError(
                "BubbleSystemBootstrap could not resolve ISpeechRecognitionService. " +
                "Ensure GlobalGameServicesBootstrap is present with realtimeWhisper assigned.",
                this);

            return null;
        }

        private MonoBehaviour FindFirstBehaviourImplementing<T>(Component excludedComponent) where T : class
        {
            var localBehaviours = GetComponents<MonoBehaviour>();
            for (int i = 0; i < localBehaviours.Length; i++)
            {
                var behaviour = localBehaviours[i];
                if (behaviour == null || behaviour == excludedComponent)
                {
                    continue;
                }

                if (behaviour is T)
                {
                    return behaviour;
                }
            }

            var sceneBehaviours = FindObjectsOfType<MonoBehaviour>(true);
            for (int i = 0; i < sceneBehaviours.Length; i++)
            {
                var behaviour = sceneBehaviours[i];
                if (behaviour == null || behaviour == excludedComponent)
                {
                    continue;
                }

                if (behaviour is T)
                {
                    return behaviour;
                }
            }

            return null;
        }
    }
}