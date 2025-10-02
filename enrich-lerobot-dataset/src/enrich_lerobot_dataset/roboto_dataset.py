import roboto


def create_roboto_dataset(
    derived_from: str | None, org_id: str | None
) -> roboto.Dataset:
    desc = "Derived from LeRobot dataset"
    if derived_from is not None:
        desc = f"{desc} from Roboto dataset {derived_from}"

    return roboto.Dataset.create(desc, tags=["derivative"], caller_org_id=org_id)
