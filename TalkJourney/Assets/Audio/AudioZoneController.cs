using UnityEngine;
using System.Collections;
using System;

public class AudioZoneController : MonoBehaviour
{
    [Header("Audio Settings")]
    public float maxVolume = 1.0f;
    public float fadeTime = 0.8f;

    // Exposes zone state transitions for external systems.
    public event Action OnPlayerEnteredBuilding;
    public event Action OnPlayerExitedBuilding;

    private AudioSource[] zoneAudios;
    
    // Caches initial inspector values to preserve the relative audio mix during scaled transitions.
    private float[] originalVolumes;
    
    private Coroutine fadeCoroutine;
    private int activeTriggers = 0;
    private string playerTag = "Player"; 

    void Awake()
    {
        zoneAudios = GetComponentsInChildren<AudioSource>();
        originalVolumes = new float[zoneAudios.Length];

        for (int i = 0; i < zoneAudios.Length; i++)
        {
            originalVolumes[i] = zoneAudios[i].volume;
            zoneAudios[i].volume = 0f;
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag(playerTag))
        {
            activeTriggers++;
            if (activeTriggers == 1)
            {
                if (fadeCoroutine != null) StopCoroutine(fadeCoroutine);
                fadeCoroutine = StartCoroutine(FadeAudio(maxVolume));
                
                OnPlayerEnteredBuilding?.Invoke();
            }
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other.CompareTag(playerTag))
        {
            activeTriggers--;
            if (activeTriggers < 0) activeTriggers = 0;

            if (activeTriggers == 0)
            {
                if (fadeCoroutine != null) StopCoroutine(fadeCoroutine);
                fadeCoroutine = StartCoroutine(FadeAudio(0f));
                
                OnPlayerExitedBuilding?.Invoke();
            }            
        }
    }

    private IEnumerator FadeAudio(float targetScale)
    {
        float[] startingVolumes = new float[zoneAudios.Length];
        for (int i = 0; i < zoneAudios.Length; i++)
        {
            startingVolumes[i] = zoneAudios[i].volume;
        }

        float currentTime = 0f;
        while (currentTime < fadeTime)
        {
            currentTime += Time.deltaTime;
            float t = currentTime / fadeTime;

            for (int i = 0; i < zoneAudios.Length; i++)
            {
                // Calculates the proportional target to maintain the intended mix ratio.
                float targetVolume = targetScale * originalVolumes[i];
                zoneAudios[i].volume = Mathf.Lerp(startingVolumes[i], targetVolume, t);
            }
            yield return null;
        }

        for (int i = 0; i < zoneAudios.Length; i++)
        {
            zoneAudios[i].volume = targetScale * originalVolumes[i];
        }
    }
}