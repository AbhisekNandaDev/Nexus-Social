# Import all models here so that Base.metadata is fully populated
# before create_all() is called at startup.
from src.db.models.auth import RefreshToken
from src.db.models.cluster import ClusterCentroid
from src.db.models.post import Post, PostEmbedding, PostFrameResult
from src.db.models.social import Follow, Like
from src.db.models.user import User, UserInterestProfile, UserPreference

__all__ = [
    "User",
    "UserPreference",
    "UserInterestProfile",
    "Post",
    "PostEmbedding",
    "PostFrameResult",
    "ClusterCentroid",
    "Follow",
    "Like",
    "RefreshToken",
]
