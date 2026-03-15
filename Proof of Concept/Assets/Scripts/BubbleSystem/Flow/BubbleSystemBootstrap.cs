using UnityEngine;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Bubbles;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Localization;
using TalkJourney.BubbleSystem.Speech;

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

        [Header("External Services")]
        [Tooltip("Optional speech recognition component implementing ISpeechRecognitionService.")]
        public MonoBehaviour speechRecognitionBehaviour;

        [Tooltip("Optional localization component implementing ILocalizationService. If empty, LocalizationResolver is added here.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Tooltip("Optional audio backend component implementing IAudioBackendClient. If empty, AudioBackendClient is added here.")]
        public MonoBehaviour audioBackendClientBehaviour;

        [Tooltip("Optional playback source. If empty, an AudioSource is added here.")]
        public AudioSource playbackSource;

        [Header("Auto Setup")]
        public bool autoFindSpeechRecognitionBehaviour = true;
        public bool autoCreateAudioSource = true;

        private void Reset()
        {
            EnsureSetup();
        }

        private void Awake()
        {
            EnsureSetup();
        }

        public void EnsureSetup()
        {
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

            if (speechRecognitionBehaviour == null && autoFindSpeechRecognitionBehaviour)
            {
                speechRecognitionBehaviour = FindFirstBehaviourImplementing<ISpeechRecognitionService>(selectionSpeechMatcher);
            }

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
            selectionSpeechMatcher.RefreshDependencies();

            audioPlaybackManager.playbackSource = playbackSource;
            audioPlaybackManager.backendClientBehaviour = audioBackendClientBehaviour;
            audioPlaybackManager.RefreshDependencies();
        }

        private MonoBehaviour EnsureLocalizationService()
        {
            if (localizationServiceBehaviour != null)
            {
                return localizationServiceBehaviour;
            }

            localizationServiceBehaviour = GetComponent<LocalizationResolver>();
            if (localizationServiceBehaviour == null)
            {
                localizationServiceBehaviour = gameObject.AddComponent<LocalizationResolver>();
            }

            return localizationServiceBehaviour;
        }

        private MonoBehaviour EnsureAudioBackendClient()
        {
            if (audioBackendClientBehaviour != null)
            {
                return audioBackendClientBehaviour;
            }

            audioBackendClientBehaviour = GetComponent<AudioBackendClient>();
            if (audioBackendClientBehaviour == null)
            {
                audioBackendClientBehaviour = gameObject.AddComponent<AudioBackendClient>();
            }

            return audioBackendClientBehaviour;
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