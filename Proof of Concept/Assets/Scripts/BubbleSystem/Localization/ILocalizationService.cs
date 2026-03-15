namespace TalkJourney.BubbleSystem.Localization
{
    public interface ILocalizationService
    {
        string Resolve(string key);
        bool TryResolve(string key, out string localizedValue);

        string ResolveForLocaleCode(string key, string localeCode);
        bool TryResolveForLocaleCode(string key, string localeCode, out string localizedValue);
    }
}
