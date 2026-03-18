using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Localization;

namespace TalkJourney.BubbleSystem.Bubbles
{
    public enum SentencePlaybackMode
    {
        SequentialBubbles,
        PreferFullSentenceClip
    }

    [DisallowMultipleComponent]
    public class SentenceBubbleController : MonoBehaviour
    {
        [Header("Dependencies")]
        [Tooltip("Component implementing IAudioPlaybackManager.")]
        public MonoBehaviour audioPlaybackManagerBehaviour;

        [Tooltip("Component implementing ILocalizationService; injected into spawned display bubbles.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Header("Composition")]
        [Tooltip("Parent transform where display bubbles are instantiated.")]
        public Transform sentenceBubbleParent;

        [Tooltip("If enabled, sentenceBubbleParent will be configured to use FlowLayoutGroup at runtime.")]
        public bool autoConfigureSentenceAreaLayout = true;

        [Tooltip("Fallback prefab if BubbleData.visualElementPrefab is not set.")]
        public DisplayBubbleController fallbackDisplayBubblePrefab;

        [Header("Speaker Button")]
        [Tooltip("Interactable speaker button shown near the sentence bubble.")]
        public VrPointerInteractable speakerButtonInteractable;

        public SentencePlaybackMode playbackMode = SentencePlaybackMode.PreferFullSentenceClip;

        private readonly List<DisplayBubbleController> _spawnedBubbles = new List<DisplayBubbleController>();
        private IAudioPlaybackManager _audioPlaybackManager;
        private StageData _activeStage;

        private void Awake()
        {
            RefreshDependencies();
            // EnsureSentenceAreaLayout();
        }

        private void OnEnable()
        {
            if (speakerButtonInteractable != null)
            {
                speakerButtonInteractable.Clicked += OnSpeakerClicked;
            }

            // Subscribe to language change events to refresh bubbles when language switches
            LocalizationResolver.OnLanguageChanged += RefreshBubblesForLanguageChange;
        }

        private void OnDisable()
        {
            if (speakerButtonInteractable != null)
            {
                speakerButtonInteractable.Clicked -= OnSpeakerClicked;
            }

            // Unsubscribe from language change events
            LocalizationResolver.OnLanguageChanged -= RefreshBubblesForLanguageChange;
        }

        private void RefreshBubblesForLanguageChange()
        {
            // Refresh all spawned display bubbles when language changes
            foreach (var bubble in _spawnedBubbles)
            {
                if (bubble != null)
                {
                    bubble.RefreshLocalizedTexts();
                }
            }
        }

        public void LoadStage(StageData stageData)
        {
            _activeStage = stageData;
            RebuildSentence(stageData);
        }

        public void RebuildSentence(StageData stageData)
        {
            ClearSpawnedBubbles();

            // EnsureSentenceAreaLayout();

            if (stageData == null || sentenceBubbleParent == null)
            {
                return;
            }

            for (int i = 0; i < stageData.sentenceBubbles.Count; i++)
            {
                var bubbleData = stageData.sentenceBubbles[i];
                if (bubbleData == null)
                {
                    continue;
                }

                var controller = SpawnBubbleController(bubbleData);
                if (controller == null)
                {
                    continue;
                }

                controller.Initialize(bubbleData);
                _spawnedBubbles.Add(controller);
            }

            var sentenceRect = sentenceBubbleParent as RectTransform;
            if (sentenceRect != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(sentenceRect);
            }
        }

        public void RefreshDependencies()
        {
            _audioPlaybackManager = audioPlaybackManagerBehaviour as IAudioPlaybackManager;
            if (_audioPlaybackManager == null)
            {
                Debug.LogError("SentenceBubbleController requires audioPlaybackManagerBehaviour implementing IAudioPlaybackManager.", this);
            }
        }

        public async Task PlaySentenceAsync(CancellationToken cancellationToken = default)
        {
            if (_audioPlaybackManager == null || _activeStage == null)
            {
                return;
            }

            if (playbackMode == SentencePlaybackMode.PreferFullSentenceClip
                && !string.IsNullOrWhiteSpace(_activeStage.fullSentenceAudioIdentifier))
            {
                var played = await _audioPlaybackManager.PlayByIdentifierAsync(_activeStage.fullSentenceAudioIdentifier, cancellationToken);
                if (played)
                {
                    return;
                }
            }

            var identifiers = new List<string>(_activeStage.sentenceBubbles.Count);
            for (int i = 0; i < _activeStage.sentenceBubbles.Count; i++)
            {
                var bubble = _activeStage.sentenceBubbles[i];
                if (bubble != null && !string.IsNullOrWhiteSpace(bubble.audioIdentifier))
                {
                    identifiers.Add(bubble.audioIdentifier);
                }
            }

            await _audioPlaybackManager.PlaySequenceAsync(identifiers, cancellationToken);
        }

        private void OnSpeakerClicked()
        {
            _ = PlaySentenceAsync();
        }

        private DisplayBubbleController SpawnBubbleController(BubbleData bubbleData)
        {
            var prefab = ResolvePrefab(bubbleData);
            if (prefab == null)
            {
                return null;
            }

            var instance = Instantiate(prefab, sentenceBubbleParent);
            instance.localizationServiceBehaviour = localizationServiceBehaviour;
            instance.audioPlaybackManagerBehaviour = audioPlaybackManagerBehaviour;
            instance.RefreshDependencies();
            return instance;
        }

        private DisplayBubbleController ResolvePrefab(BubbleData bubbleData)
        {
            if (bubbleData != null && bubbleData.visualElementPrefab != null)
            {
                return bubbleData.visualElementPrefab.GetComponent<DisplayBubbleController>();
            }

            return fallbackDisplayBubblePrefab;
        }

        private void ClearSpawnedBubbles()
        {
            for (int i = 0; i < _spawnedBubbles.Count; i++)
            {
                var spawned = _spawnedBubbles[i];
                if (spawned != null)
                {
                    Destroy(spawned.gameObject);
                }
            }

            _spawnedBubbles.Clear();
        }
    }
}
