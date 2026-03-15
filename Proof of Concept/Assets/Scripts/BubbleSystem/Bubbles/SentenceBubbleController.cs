using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Layout;
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
            EnsureSentenceAreaLayout();
        }

        private void OnEnable()
        {
            if (speakerButtonInteractable != null)
            {
                speakerButtonInteractable.Clicked += OnSpeakerClicked;
            }
        }

        private void OnDisable()
        {
            if (speakerButtonInteractable != null)
            {
                speakerButtonInteractable.Clicked -= OnSpeakerClicked;
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

            EnsureSentenceAreaLayout();

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

        private void EnsureSentenceAreaLayout()
        {
            if (!autoConfigureSentenceAreaLayout || sentenceBubbleParent == null)
            {
                return;
            }

            var sentenceRect = sentenceBubbleParent as RectTransform;
            if (sentenceRect == null)
            {
                return;
            }

            var flowLayout = sentenceRect.GetComponent<FlowLayoutGroup>();
            if (flowLayout == null)
            {
                flowLayout = sentenceRect.gameObject.AddComponent<FlowLayoutGroup>();
                flowLayout.childAlignment = TextAnchor.UpperLeft;
                flowLayout.horizontalSpacing = 12f;
                flowLayout.verticalSpacing = 12f;
            }

            var horizontalLayout = sentenceRect.GetComponent<HorizontalLayoutGroup>();
            if (horizontalLayout != null)
            {
                horizontalLayout.enabled = false;
            }

            if (sentenceRect.rect.width <= 1f)
            {
                Debug.LogWarning("SentenceArea width is zero or too small. Set a fixed width in RectTransform so wrapping can work.", sentenceRect);
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
