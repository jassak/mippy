from addict import Dict


properties = Dict(
    {
        "name": "Logistic Regression",
        "parameters": {
            "columns": {
                "target": {
                    "names": ["alzheimerbroadcategory"],
                    "required": True,
                    "types": ["categorical"],
                },
                "features": {
                    "names": ["lefthippocampus"],
                    "required": True,
                    "types": ["numerical"],
                },
            },
            "dataset": ["adni"],
            "filter": {"alzheimerbroadcategory": ["CN", "AD"]}
        },
    }
)
