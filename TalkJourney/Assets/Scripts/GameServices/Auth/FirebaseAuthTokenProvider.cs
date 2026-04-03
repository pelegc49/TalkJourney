using System;
using System.Threading;
using System.Threading.Tasks;
using Firebase;
using Firebase.Auth;
using UnityEngine;

namespace TalkJourney.GameServices.Auth
{
    [DisallowMultipleComponent]
    public class FirebaseAuthTokenProvider : MonoBehaviour, IAuthTokenProvider
    {
        [Tooltip("If true, refreshes Firebase token before each request.")]
        public bool forceRefreshFirebaseToken = true;

        [Tooltip("If enabled, performs anonymous sign-in when no Firebase user is available.")]
        public bool signInAnonymouslyIfNeeded = true;

        public async Task<string> GetAuthorizationTokenAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                var dependencyStatus = await FirebaseApp.CheckAndFixDependenciesAsync();
                if (dependencyStatus != DependencyStatus.Available)
                {
                    Debug.LogWarning($"Firebase dependencies unavailable: {dependencyStatus}", this);
                    return null;
                }

                if (cancellationToken.IsCancellationRequested)
                {
                    return null;
                }

                var auth = FirebaseAuth.DefaultInstance;
                var user = auth.CurrentUser;

                if (user == null && signInAnonymouslyIfNeeded)
                {
                    var signInResult = await auth.SignInAnonymouslyAsync();
                    user = signInResult?.User;
                }

                if (user == null)
                {
                    return null;
                }

                if (cancellationToken.IsCancellationRequested)
                {
                    return null;
                }

                return await user.TokenAsync(forceRefreshFirebaseToken);
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"Firebase token fetch failed: {exception.Message}", this);
                return null;
            }
        }
    }
}
