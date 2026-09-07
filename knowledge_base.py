# knowledge_base.py
"""
Task 2: 12 required knowledge-base documents, 2-5 sentences each,
covering every required topic in the scenario.
"""

DOCUMENTS = {
    "doc_1": "Personal Loan Eligibility: Applicants require a minimum monthly net income of 25,000 INR and a CIBIL score of 700 or higher. Business loans require at least 3 years of audited operational history and an annual turnover exceeding 50 Lakhs INR.",
    "doc_2": "EMI Calculation Rules: Equated Monthly Installments (EMIs) are calculated using the reducing balance method. The interest is applied solely to the outstanding loan principal remaining after every preceding installment.",
    "doc_3": "Credit Card Fee Structure: Standard credit cards carry an annual maintenance fee of 500 INR, which is waived if annual retail spends exceed 1 Lakh INR. Late payment charges start at 250 INR for outstanding balances over 10,000 INR.",
    "doc_4": "KYC Document Requirements: Mandatory identity verification requires a valid PAN card, alongside address verification through an Aadhaar card or Passport. Video KYC requires displaying the original physical PAN to the agent.",
    "doc_5": "Fraud-Dispute Resolution Process: Suspected fraudulent card charges must be registered within 3 calendar days of occurrence. The risk operations team provides a provisional credit within 5 business days while investigation concludes.",
    "doc_6": "Account-Closure Process: Accounts can only be closed once all overdrafts, active loans, and maintenance fees are settled to zero. A closure ticket can be filed through the mobile banking app and completes within 7 business days.",
    "doc_7": "Interest-Rate Slabs: Savings deposits up to 1 Lakh INR accrue 3.0% annual interest. Balances from 1 Lakh to 10 Lakhs INR accrue 4.0%, and balances exceeding 10 Lakhs INR yield 5.0% calculated quarterly.",
    "doc_8": "Prepayment-Penalty Rules: Auto loans settled within 12 months of sanction incur a 2% prepayment surcharge on the outstanding balance. Home loans issued to individuals under floating interest rates carry zero prepayment penalty.",
    "doc_9": "Minimum-Balance Requirements: Corporate payroll salary accounts require zero minimum balance. Standard retail savings accounts mandate an Average Monthly Balance (AMB) of 10,000 INR, with a 300 INR penalty if breached.",
    "doc_10": "Credit-Score Impact Factors: Repayment punctuality comprises 35% of the overall credit rating, while credit utilization ratio accounts for 30%. Excessive hard inquiries submitted within a short interval temporarily suppress the score.",
    "doc_11": "Joint-Account Rules: Primary and secondary holders possess equal debit access to deposited funds and share joint liability for liabilities. Deleting or modifying a joint holder requires explicit physical consent from both account holders.",
    "doc_12": "NRI-Account Eligibility: Non-Resident Indians can initiate NRE or NRO accounts with an Indian passport, valid overseas visa, and foreign address proof. Initial remittances must originate exclusively from foreign banking channels.",
}

# Topic label for each doc, used by eval.py to guarantee every required
# topic gets at least one test query in the 15-query evaluation set.
DOC_TOPICS = {
    "doc_1": "loan eligibility criteria by loan type",
    "doc_2": "EMI calculation rules",
    "doc_3": "credit-card fee structure",
    "doc_4": "KYC document requirements",
    "doc_5": "fraud-dispute resolution process",
    "doc_6": "account-closure process",
    "doc_7": "interest-rate slabs",
    "doc_8": "prepayment-penalty rules",
    "doc_9": "minimum-balance requirements",
    "doc_10": "credit-score impact factors",
    "doc_11": "joint-account rules",
    "doc_12": "NRI-account eligibility",
}
