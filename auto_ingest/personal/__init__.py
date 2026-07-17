"""Personal 'memory recall' feature: embed + link personal photos/videos.

Personal iPhone media is NOT content; these modules embed media with CLIP and
link :MediaFile nodes to :SummaryPlace / :Trip / :PhoneLog via GPS haversine.
"""

__all__ = [
    "embed",
    "link_media",
    "recall",
]
