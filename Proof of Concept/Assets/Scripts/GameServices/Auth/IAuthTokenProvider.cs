using System.Threading;
using System.Threading.Tasks;

namespace TalkJourney.GameServices.Auth
{
    public interface IAuthTokenProvider
    {
        Task<string> GetAuthorizationTokenAsync(CancellationToken cancellationToken = default);
    }
}
