"""Dev-only JWT issuer (stands in for the company IdP).

Usage: python -m scripts.issue_token --company nimbus|kestrel|saffron --sub emp1 [--role admin]
"""

import argparse

from app.auth import issue_token
from app.config import get_settings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--company", required=True, choices=sorted(get_settings().companies))
    p.add_argument("--sub", required=True)
    p.add_argument("--role", default="employee", choices=["employee", "admin"])
    args = p.parse_args()
    print(issue_token(sub=args.sub, company_id=args.company, role=args.role))


if __name__ == "__main__":
    main()
