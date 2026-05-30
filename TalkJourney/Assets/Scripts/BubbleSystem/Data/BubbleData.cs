using System;
using UnityEngine;

namespace TalkJourney.BubbleSystem.Data
{
    public enum BubbleVisualType
    {
        Text,
        Image
    }

    [Serializable]
    public class BubbleData
    {
        [Header("Localization")]
        [Tooltip("Single localization key used for both primary text and transliteration lookup.")]
        public string primaryTextKey;

        [Tooltip("Direct primary text used when the bubble is created from resolved sentence text instead of a localization key.")]
        public string primaryTextOverride;

        [Tooltip("Direct transliteration text used when the bubble is created from resolved sentence text instead of a localization key.")]
        public string transliteratorTextOverride;

        [Header("Audio")]
        [Tooltip("Unique ID or endpoint suffix used by backend audio service.")]
        public string audioIdentifier;

        [Header("Visual")]
        [Tooltip("Determines whether this bubble binds to a Text-like or Image-like visual.")]
        public BubbleVisualType visualType = BubbleVisualType.Text;

        [Tooltip("Prefab that contains the visual element for this bubble.")]
        public GameObject visualElementPrefab;
    }
}
