using UnityEngine;
using System.Collections;

public class OutsideAmbienceManager : MonoBehaviour
{
    [Header("Interior Zones")]
    public AudioZoneController[] interiorZones;

    [Header("Outside Audio Settings")]
    public float maxVolume = 1.0f;
    public float fadeTime = 0.8f;

    private AudioSource[] outsideAudios;
    
    // Caches inspector volume settings to preserve the relative audio mix ratio during global volume scaling.
    private float[] originalVolumes;
    
    private Coroutine fadeCoroutine;
    private int activeInteriorZones = 0; 

    void Awake()
    {
        outsideAudios = GetComponentsInChildren<AudioSource>();
        originalVolumes = new float[outsideAudios.Length];

        for (int i = 0; i < outsideAudios.Length; i++)
        {
            originalVolumes[i] = outsideAudios[i].volume;
            
            // Initializes the starting volume based on the global maxVolume multiplier.
            outsideAudios[i].volume = originalVolumes[i] * maxVolume;
        }
    }

    void OnEnable()
    {
        foreach (var zone in interiorZones)
        {
            if (zone != null)
            {
                zone.OnPlayerEnteredBuilding += HandlePlayerEnteredBuilding;
                zone.OnPlayerExitedBuilding += HandlePlayerExitedBuilding;
            }
        }
    }

    void OnDisable()
    {
        // Ensures deterministic event unsubscription to prevent memory leaks during object destruction.
        foreach (var zone in interiorZones)
        {
            if (zone != null)
            {
                zone.OnPlayerEnteredBuilding -= HandlePlayerEnteredBuilding;
                zone.OnPlayerExitedBuilding -= HandlePlayerExitedBuilding;
            }
        }
    }

    private void HandlePlayerEnteredBuilding()
    {
        activeInteriorZones++;
        
        // Initiates audio suppression upon entry into the first interior zone bounds.
        if (activeInteriorZones == 1)
        {
            if (fadeCoroutine != null) StopCoroutine(fadeCoroutine);
            fadeCoroutine = StartCoroutine(FadeAudio(0f));
        }
    }

    private void HandlePlayerExitedBuilding()
    {
        activeInteriorZones--;
        
        // Failsafe constraint to maintain state integrity.
        if (activeInteriorZones < 0) activeInteriorZones = 0;

        // Restores ambient audio upon complete exit from all overlapping interior zones.
        if (activeInteriorZones == 0)
        {
            if (fadeCoroutine != null) StopCoroutine(fadeCoroutine);
            fadeCoroutine = StartCoroutine(FadeAudio(maxVolume));
        }
    }

    private IEnumerator FadeAudio(float targetScale)
    {
        float[] startingVolumes = new float[outsideAudios.Length];
        for (int i = 0; i < outsideAudios.Length; i++)
        {
            startingVolumes[i] = outsideAudios[i].volume;
        }

        float currentTime = 0f;
        while (currentTime < fadeTime)
        {
            currentTime += Time.deltaTime;
            float t = currentTime / fadeTime;

            for (int i = 0; i < outsideAudios.Length; i++)
            {
                // Calculates the proportional target volume relative to the original mix parameters.
                float targetVolume = targetScale * originalVolumes[i];
                outsideAudios[i].volume = Mathf.Lerp(startingVolumes[i], targetVolume, t);
            }
            yield return null;
        }

        for (int i = 0; i < outsideAudios.Length; i++)
        {
            outsideAudios[i].volume = targetScale * originalVolumes[i];
        }
    }
}