using UnityEngine;
using UnityEngine.Audio;
using UnityEngine.UI;

public class VolumeSettings : MonoBehaviour
{

    [SerializeField]
    private AudioMixer mixer;
    [SerializeField]
    private Slider masterSlider;
    [SerializeField]
    private Slider environmentSlider;
    [SerializeField]
    private Slider voiceSlider;
    [SerializeField]
    private Slider SFXSlider;
    [SerializeField]
    private Slider musicSlider;

    public void SetMusicVolume()
    {
        float volume = musicSlider.value;
        mixer.SetFloat("music", Mathf.Log10(volume)*20);
        PlayerPrefs.SetFloat("musicVolume", volume);
    }

    public void SetSFXVolume()
    {
        float volume = SFXSlider.value;
        mixer.SetFloat("sfx", Mathf.Log10(volume)*20);
        PlayerPrefs.SetFloat("sfxVolume", volume);
    }

    public void SetMasterVolume()
    {
        float volume = masterSlider.value;
        mixer.SetFloat("master", Mathf.Log10(volume)*20);
        PlayerPrefs.SetFloat("masterVolume", volume);
    }

    public void SetEnvironmentVolume()
    {
        float volume = environmentSlider.value;
        mixer.SetFloat("environment", Mathf.Log10(volume)*20);
        PlayerPrefs.SetFloat("environmentVolume", volume);
    }

    public void SetVoiceVolume()
    {
        float volume = voiceSlider.value;
        mixer.SetFloat("voice", Mathf.Log10(volume)*20);
        PlayerPrefs.SetFloat("voiceVolume", volume);
    }

    public void LoadVolume()
    {
        if (PlayerPrefs.HasKey("masterVolume"))
            masterSlider.value = PlayerPrefs.GetFloat("masterVolume");
        if (PlayerPrefs.HasKey("environmentVolume"))
            environmentSlider.value = PlayerPrefs.GetFloat("environmentVolume");
        if (PlayerPrefs.HasKey("voiceVolume"))
            voiceSlider.value = PlayerPrefs.GetFloat("voiceVolume");
        if (PlayerPrefs.HasKey("sfxVolume"))
            SFXSlider.value = PlayerPrefs.GetFloat("sfxVolume");
        if (PlayerPrefs.HasKey("musicVolume"))
            musicSlider.value = PlayerPrefs.GetFloat("musicVolume");

        SetMusicVolume();
        SetSFXVolume();
        SetMasterVolume();
        SetEnvironmentVolume();
        SetVoiceVolume();
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        LoadVolume();
    }
}
