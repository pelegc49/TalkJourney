using System;
using System.Collections.Generic;
using UnityEngine;
using TalkJourney.BubbleSystem.Bubbles;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Speech;

namespace TalkJourney.BubbleSystem.Flow
{
    [DisallowMultipleComponent]
    public class StageController : MonoBehaviour, IStageController
    {
        [Header("Stage")]
        public StageData initialStage;

        [Header("Sentence")]
        public SentenceBubbleController sentenceBubbleController;

        [Header("Selection")]
        [Tooltip("Parent transform where selection bubbles are instantiated.")]
        public Transform selectionBubbleParent;

        [Tooltip("Prefab used to instantiate each selection bubble.")]
        public SelectionBubbleController selectionBubblePrefab;

        [Tooltip("Local-space origin used for selection bubble layout.")]
        public Vector3 selectionBubbleLayoutOrigin = Vector3.zero;

        [Tooltip("Horizontal spacing (local X axis) between selection bubbles.")]
        public float selectionBubbleHorizontalSpacing = 0.6f;

        [Tooltip("When enabled, bubbles are centered around the layout origin.")]
        public bool centerSelectionBubbleLayout = true;

        [Header("Speech")]
        public SelectionSpeechMatcher selectionSpeechMatcher;

        [Header("Dependency Injection")]
        [Tooltip("Component implementing localization service; injected into spawned selection bubble display controllers.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Tooltip("Component implementing audio playback manager; injected into spawned selection bubble display controllers.")]
        public MonoBehaviour audioPlaybackManagerBehaviour;

        public StageData CurrentStage { get; private set; }
        public event Action<StageData> StageChanged;

        private readonly List<SelectionBubbleController> _activeSelectionBubbles = new List<SelectionBubbleController>();

        private void Start()
        {
            if (initialStage != null)
            {
                LoadStage(initialStage);
            }
        }

        public bool TransitionToStage(StageData nextStage)
        {
            return LoadStage(nextStage);
        }

        public bool LoadStage(StageData nextStage)
        {
            if (nextStage == null)
            {
                Debug.LogWarning("Cannot load null stage.", this);
                return false;
            }

            CurrentStage = nextStage;

            if (sentenceBubbleController != null)
            {
                sentenceBubbleController.LoadStage(nextStage);
            }

            RebuildSelectionBubbles(nextStage);
            StageChanged?.Invoke(nextStage);
            BubbleEventBus.PublishStageChanged(nextStage);
            return true;
        }

        private void RebuildSelectionBubbles(StageData stageData)
        {
            ClearSelectionBubbles();

            if (selectionBubbleParent == null || selectionBubblePrefab == null || stageData == null)
            {
                PushActiveSelectionsToSpeechMatcher();
                return;
            }

            for (int i = 0; i < stageData.selectionBubbles.Count; i++)
            {
                var selectionData = stageData.selectionBubbles[i];
                if (selectionData == null)
                {
                    continue;
                }

                var instance = Instantiate(selectionBubblePrefab, selectionBubbleParent);
                instance.selectionData = selectionData;
                instance.stageControllerBehaviour = this;
                instance.localizationServiceBehaviour = localizationServiceBehaviour;
                instance.RefreshDependencies();
                _activeSelectionBubbles.Add(instance);

                var displayBubble = instance.GetComponent<DisplayBubbleController>();
                if (displayBubble != null)
                {
                    displayBubble.bubbleData = selectionData.bubble;
                    displayBubble.localizationServiceBehaviour = localizationServiceBehaviour;
                    displayBubble.audioPlaybackManagerBehaviour = audioPlaybackManagerBehaviour;
                    displayBubble.RefreshDependencies();
                    displayBubble.RefreshLocalizedTexts();
                }
            }

            ArrangeSelectionBubbles();

            PushActiveSelectionsToSpeechMatcher();
        }

        private void ArrangeSelectionBubbles()
        {
            if (_activeSelectionBubbles.Count == 0)
            {
                return;
            }

            var midpoint = centerSelectionBubbleLayout ? (_activeSelectionBubbles.Count - 1) * 0.5f : 0f;

            for (int i = 0; i < _activeSelectionBubbles.Count; i++)
            {
                var bubble = _activeSelectionBubbles[i];
                if (bubble == null)
                {
                    continue;
                }

                var localPosition = selectionBubbleLayoutOrigin;
                localPosition.x += (i - midpoint) * selectionBubbleHorizontalSpacing;
                bubble.transform.localPosition = localPosition;
            }
        }

        private void PushActiveSelectionsToSpeechMatcher()
        {
            if (selectionSpeechMatcher != null)
            {
                selectionSpeechMatcher.SetActiveSelectionBubbles(_activeSelectionBubbles);
            }
        }

        private void ClearSelectionBubbles()
        {
            for (int i = 0; i < _activeSelectionBubbles.Count; i++)
            {
                var bubble = _activeSelectionBubbles[i];
                if (bubble != null)
                {
                    Destroy(bubble.gameObject);
                }
            }

            _activeSelectionBubbles.Clear();
        }
    }
}
