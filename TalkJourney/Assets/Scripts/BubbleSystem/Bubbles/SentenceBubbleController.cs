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

        [Tooltip("Optional Speaker child GameObject. If empty, a child named 'Speaker' is used.")]
        public GameObject speakerIconObject;

        [Tooltip("Optional Loading child GameObject. If empty, a child named 'Loading' is used.")]
        public GameObject loadingIconObject;

        [Tooltip("If enabled, speaker button becomes a child of sentenceBubbleParent and participates in its layout flow.")]
        public bool includeSpeakerInSentenceLayout = true;

        public SentencePlaybackMode playbackMode = SentencePlaybackMode.PreferFullSentenceClip;

        private readonly List<DisplayBubbleController> _spawnedBubbles = new List<DisplayBubbleController>();
        private readonly List<BubbleData> _activeSentenceBubbleData = new List<BubbleData>();
        private IAudioPlaybackManager _audioPlaybackManager;
        private ILocalizationService _localizationService;
        private StageData _activeStage;
        private Coroutine _pendingSentenceRebuild;

        private void Awake()
        {
            RefreshDependencies();
            // EnsureSentenceAreaLayout();
            SetSpeakerButtonVisible(false);
            SetSpeakerPlaybackState(false);
        }

        private void OnEnable()
        {
            if (speakerButtonInteractable != null)
            {
                speakerButtonInteractable.Clicked += OnSpeakerClicked;
            }

            // Subscribe to language change events to refresh bubbles when language switches
            LocalizationResolver.OnLanguageChanged += RefreshBubblesForLanguageChange;
            LocalizationResolver.OnTransliteratorChanged += RefreshBubblesForLanguageChange;
        }

        private void OnDisable()
        {
            if (speakerButtonInteractable != null)
            {
                speakerButtonInteractable.Clicked -= OnSpeakerClicked;
            }

            // Unsubscribe from language change events
            LocalizationResolver.OnLanguageChanged -= RefreshBubblesForLanguageChange;
            LocalizationResolver.OnTransliteratorChanged -= RefreshBubblesForLanguageChange;

            if (_pendingSentenceRebuild != null)
            {
                StopCoroutine(_pendingSentenceRebuild);
                _pendingSentenceRebuild = null;
            }
        }

        private void RefreshBubblesForLanguageChange()
        {
            if (_activeStage != null && _pendingSentenceRebuild == null)
            {
                _pendingSentenceRebuild = StartCoroutine(RebuildSentenceAfterLocalizationChange());
            }
        }

        private System.Collections.IEnumerator RebuildSentenceAfterLocalizationChange()
        {
            yield return null;

            _pendingSentenceRebuild = null;

            if (_activeStage != null)
            {
                RebuildSentence(_activeStage);
            }
        }

        public void LoadStage(StageData stageData)
        {
            _activeStage = stageData;
            RebuildSentence(stageData);

            var shouldShowSpeaker = stageData != null;
            SetSpeakerButtonVisible(shouldShowSpeaker);
            if (shouldShowSpeaker)
            {
                SetSpeakerPlaybackState(false);
                SyncSpeakerWithSentenceLayout();
            }
        }

        public void RebuildSentence(StageData stageData)
        {
            ClearSpawnedBubbles();
            _activeSentenceBubbleData.Clear();

            // EnsureSentenceAreaLayout();

            ApplySentenceLayoutDirection();

            if (stageData == null || sentenceBubbleParent == null)
            {
                return;
            }

            var sentenceBubbleData = BuildSentenceBubbleData(stageData);
            for (int i = 0; i < sentenceBubbleData.Count; i++)
            {
                var bubbleData = sentenceBubbleData[i];
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
                _activeSentenceBubbleData.Add(bubbleData);
            }

            var sentenceRect = sentenceBubbleParent as RectTransform;
            if (sentenceRect != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(sentenceRect);
            }

            SyncSpeakerWithSentenceLayout();
        }

        /// <summary>
        /// Clears currently spawned sentence bubbles and hides sentence controls.
        /// </summary>
        public void ClearStageVisuals()
        {
            _activeStage = null;
            ClearSpawnedBubbles();
            SetSpeakerButtonVisible(false);
        }

        public void RefreshDependencies()
        {
            _audioPlaybackManager = audioPlaybackManagerBehaviour as IAudioPlaybackManager;
            _localizationService = localizationServiceBehaviour as ILocalizationService;

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

            var sentenceTexts = BuildSentenceTexts();
            if (sentenceTexts.Count == 0)
            {
                return;
            }

            SetSpeakerPlaybackState(true);

            try
            {
                if (playbackMode == SentencePlaybackMode.PreferFullSentenceClip)
                {
                    var fullSentenceText = string.Join(" ", sentenceTexts).Trim();
                    var played = await _audioPlaybackManager.PlayByTextAsync(fullSentenceText, cancellationToken);
                    if (played)
                    {
                        return;
                    }
                }

                await _audioPlaybackManager.PlaySequenceAsync(sentenceTexts, cancellationToken);
            }
            finally
            {
                SetSpeakerPlaybackState(false);
            }
        }

        private List<string> BuildSentenceTexts()
        {
            var texts = new List<string>(_activeSentenceBubbleData.Count);

            for (int i = 0; i < _activeSentenceBubbleData.Count; i++)
            {
                var bubble = _activeSentenceBubbleData[i];
                if (bubble == null)
                {
                    continue;
                }

                var resolved = ResolvePrimaryText(bubble).Trim();
                if (!string.IsNullOrWhiteSpace(resolved))
                {
                    texts.Add(resolved);
                }
            }

            return texts;
        }

        private string ResolveText(string key)
        {
            if (_localizationService == null || string.IsNullOrWhiteSpace(key))
            {
                return key ?? string.Empty;
            }

            return _localizationService.Resolve(key);
        }

        private string ResolvePrimaryText(BubbleData bubbleData)
        {
            if (bubbleData == null)
            {
                return string.Empty;
            }

            if (!string.IsNullOrWhiteSpace(bubbleData.primaryTextOverride))
            {
                return bubbleData.primaryTextOverride;
            }

            return ResolveText(bubbleData.primaryTextKey);
        }

        private string ResolveSentenceTransliteration(string sentenceKey, string fallbackText)
        {
            if (string.IsNullOrWhiteSpace(sentenceKey))
            {
                return fallbackText ?? string.Empty;
            }

            var transliteratorCode = GetCurrentTransliteratorCode();
            if (!string.IsNullOrWhiteSpace(transliteratorCode)
                && _localizationService != null
                && _localizationService.TryResolveForLocaleCode(sentenceKey, transliteratorCode, out var transliteratedSentence))
            {
                return transliteratedSentence;
            }

            return fallbackText ?? string.Empty;
        }

        private List<BubbleData> BuildSentenceBubbleData(StageData stageData)
        {
            var bubbleDataList = new List<BubbleData>();

            if (stageData == null)
            {
                return bubbleDataList;
            }

            if (!string.IsNullOrWhiteSpace(stageData.sentenceLocalizationKey))
            {
                var localizedSentence = ResolveText(stageData.sentenceLocalizationKey).Trim();
                if (string.IsNullOrWhiteSpace(localizedSentence))
                {
                    localizedSentence = stageData.sentenceLocalizationKey.Trim();
                }

                var transliteratedSentence = ResolveSentenceTransliteration(stageData.sentenceLocalizationKey, localizedSentence).Trim();
                bubbleDataList.AddRange(BuildWordBubbles(localizedSentence, transliteratedSentence));
                return bubbleDataList;
            }

            if (stageData.sentenceBubbles != null && stageData.sentenceBubbles.Count > 0)
            {
                bubbleDataList.AddRange(stageData.sentenceBubbles);
            }

            return bubbleDataList;
        }

        private List<BubbleData> BuildWordBubbles(string localizedSentence, string transliteratedSentence)
        {
            var displayWords = SplitSentenceIntoWords(localizedSentence);
            var transliteratedWords = SplitSentenceIntoWords(transliteratedSentence);
            var bubbleCount = displayWords.Count;

            var runtimeBubbles = new List<BubbleData>(bubbleCount);
            for (int i = 0; i < bubbleCount; i++)
            {
                var bubble = new BubbleData
                {
                    primaryTextOverride = displayWords[i],
                    transliteratorTextOverride = i < transliteratedWords.Count ? transliteratedWords[i] : displayWords[i],
                    visualType = BubbleVisualType.Text,
                    visualElementPrefab = fallbackDisplayBubblePrefab != null ? fallbackDisplayBubblePrefab.gameObject : null
                };

                runtimeBubbles.Add(bubble);
            }

            return runtimeBubbles;
        }

        private static List<string> SplitSentenceIntoWords(string sentence)
        {
            var words = new List<string>();
            if (string.IsNullOrWhiteSpace(sentence))
            {
                return words;
            }

            var segments = sentence.Split(new[] { ' ', '\t', '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < segments.Length; i++)
            {
                var segment = segments[i].Trim();
                if (!string.IsNullOrWhiteSpace(segment))
                {
                    words.Add(segment);
                }
            }

            return words;
        }

        private string GetCurrentTransliteratorCode()
        {
            if (localizationServiceBehaviour is LocalizationResolver localizationResolver)
            {
                return localizationResolver.GetCurrentTransliteratorCode();
            }

            var sceneLocalizationResolver = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
            if (sceneLocalizationResolver != null)
            {
                return sceneLocalizationResolver.GetCurrentTransliteratorCode();
            }

            return string.Empty;
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

        private void SetSpeakerButtonVisible(bool isVisible)
        {
            var speakerTransform = ResolveSpeakerButtonTransform();
            if (speakerTransform != null)
            {
                speakerTransform.gameObject.SetActive(isVisible);
            }
        }

        private void SetSpeakerPlaybackState(bool isLoading)
        {
            var speakerObject = ResolveSpeakerIconObject();
            var loadingObject = ResolveLoadingIconObject();

            if (speakerObject != null)
            {
                speakerObject.SetActive(!isLoading);
            }

            if (loadingObject != null)
            {
                loadingObject.SetActive(isLoading);
            }
        }

        private Transform ResolveSpeakerButtonTransform()
        {
            if (speakerButtonInteractable != null)
            {
                return speakerButtonInteractable.transform;
            }

            return null;
        }

        private GameObject ResolveSpeakerIconObject()
        {
            if (speakerIconObject != null)
            {
                return speakerIconObject;
            }

            var speakerTransform = ResolveSpeakerButtonTransform();
            if (speakerTransform == null)
            {
                return null;
            }

            var child = speakerTransform.Find("Speaker");
            if (child != null)
            {
                speakerIconObject = child.gameObject;
            }

            return speakerIconObject;
        }

        private GameObject ResolveLoadingIconObject()
        {
            if (loadingIconObject != null)
            {
                return loadingIconObject;
            }

            var speakerTransform = ResolveSpeakerButtonTransform();
            if (speakerTransform == null)
            {
                return null;
            }

            var child = speakerTransform.Find("Loading");
            if (child != null)
            {
                loadingIconObject = child.gameObject;
            }

            return loadingIconObject;
        }

        private RectTransform ResolveSentenceAreaRect()
        {
            if (sentenceBubbleParent == null)
            {
                return null;
            }

            return sentenceBubbleParent as RectTransform;
        }

        private void ApplySentenceLayoutDirection()
        {
            if (sentenceBubbleParent == null)
            {
                return;
            }

            var flowLayout = sentenceBubbleParent.GetComponent<FlowLayoutGroup>();
            if (flowLayout == null)
            {
                return;
            }

            var localizationResolver = ResolveLocalizationResolver();
            if (localizationResolver == null)
            {
                return;
            }

            flowLayout.rightToLeft = LocalizationResolver.IsRightToLeftLanguage(localizationResolver.learningLanguage);
            SyncSpeakerWithSentenceLayout();

            var sentenceRect = sentenceBubbleParent as RectTransform;
            if (sentenceRect != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(sentenceRect);
            }
        }

        private void SyncSpeakerWithSentenceLayout()
        {
            if (!includeSpeakerInSentenceLayout || sentenceBubbleParent == null)
            {
                return;
            }

            var speakerTransform = ResolveSpeakerButtonTransform();
            if (speakerTransform == null)
            {
                return;
            }

            if (speakerTransform.parent != sentenceBubbleParent)
            {
                speakerTransform.SetParent(sentenceBubbleParent, false);
            }

            // Keep speaker after all sentence bubbles in hierarchy order.
            speakerTransform.SetAsLastSibling();

            var sentenceRect = ResolveSentenceAreaRect();
            if (sentenceRect != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(sentenceRect);
            }
        }

        private LocalizationResolver ResolveLocalizationResolver()
        {
            var resolverFromBehaviour = localizationServiceBehaviour as LocalizationResolver;
            if (resolverFromBehaviour != null)
            {
                return resolverFromBehaviour;
            }

            return FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
        }
    }
}
