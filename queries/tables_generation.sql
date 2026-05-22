-- migrations/001_create_tenant_tables.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Organizations table (assuming this exists or we create it)
CREATE TABLE IF NOT EXISTS public.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    tenant_code VARCHAR(50) UNIQUE NOT NULL,
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Diagnostic sessions table with tenant isolation
CREATE TABLE public.diagnostic_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    session_identifier VARCHAR(100) NOT NULL,
    patient_text TEXT,
    detected_language VARCHAR(2),
    patient_age INTEGER,
    patient_gender VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint per organization
    UNIQUE(organization_id, session_identifier),
    
    -- Indexes for performance
    INDEX idx_org_sessions (organization_id),
    INDEX idx_session_identifier (session_identifier),
    INDEX idx_created_at (created_at)
);

-- Diagnostic results table with tenant isolation
CREATE TABLE public.diagnostic_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES diagnostic_sessions(id) ON DELETE CASCADE,
    condition_icd10 VARCHAR(20) NOT NULL,
    condition_name_en TEXT NOT NULL,
    condition_name_pt TEXT NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    urgency VARCHAR(20) NOT NULL,
    treatment_priority INTEGER NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    recommendation_en TEXT NOT NULL,
    recommendation_pt TEXT NOT NULL,
    risk_level_en VARCHAR(100) NOT NULL,
    risk_level_pt VARCHAR(100) NOT NULL,
    raw_input_text TEXT,
    detected_language VARCHAR(2),
    ai_model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for tenant isolation and queries
    INDEX idx_org_results (organization_id),
    INDEX idx_session_id (session_id),
    INDEX idx_org_condition (organization_id, condition_icd10),
    INDEX idx_org_created_at (organization_id, created_at),
    INDEX idx_org_severity (organization_id, severity),
    INDEX idx_org_confidence (organization_id, confidence)
);

-- Matched symptoms table (details)
CREATE TABLE public.matched_symptoms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    diagnostic_result_id UUID NOT NULL REFERENCES diagnostic_results(id) ON DELETE CASCADE,
    symptom_id VARCHAR(50) NOT NULL,
    symptom_name_en TEXT NOT NULL,
    symptom_name_pt TEXT NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    severity_weight DECIMAL(3,2),
    
    INDEX idx_org_matched_symptoms (organization_id),
    INDEX idx_diagnostic_result (diagnostic_result_id)
);

-- Differential diagnoses table
CREATE TABLE public.differential_diagnoses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    diagnostic_result_id UUID NOT NULL REFERENCES diagnostic_results(id) ON DELETE CASCADE,
    condition_icd10 VARCHAR(20) NOT NULL,
    condition_name_en TEXT NOT NULL,
    condition_name_pt TEXT NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    
    INDEX idx_org_differentials (organization_id),
    INDEX idx_diagnostic_result (diagnostic_result_id)
);

-- Required exams table
CREATE TABLE public.required_exams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    diagnostic_result_id UUID NOT NULL REFERENCES diagnostic_results(id) ON DELETE CASCADE,
    exam_name_en TEXT NOT NULL,
    exam_name_pt TEXT NOT NULL,
    priority INTEGER DEFAULT 1,
    
    INDEX idx_org_exams (organization_id),
    INDEX idx_diagnostic_result (diagnostic_result_id)
);

-- Organization-specific treatment customizations (SaaS feature)
CREATE TABLE public.organization_treatment_protocols (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    condition_icd10 VARCHAR(20) NOT NULL,
    custom_treatment_en TEXT,
    custom_treatment_pt TEXT,
    custom_medications JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(organization_id, condition_icd10),
    INDEX idx_org_treatment (organization_id, condition_icd10)
);

-- Audit log for compliance (SaaS requirement)
CREATE TABLE public.diagnostic_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    user_id UUID,
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_org_audit (organization_id),
    INDEX idx_created_at (created_at),
    INDEX idx_action (action)
);

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_sessions_updated_at 
    BEFORE UPDATE ON diagnostic_sessions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_org_treatment_updated_at 
    BEFORE UPDATE ON organization_treatment_protocols 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Row-Level Security (RLS) policies for multi-tenant isolation
ALTER TABLE public.diagnostic_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.diagnostic_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.matched_symptoms ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.differential_diagnoses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.required_exams ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_treatment_protocols ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.diagnostic_audit_log ENABLE ROW LEVEL SECURITY;

-- Create policies that ensure users only see their organization's data
CREATE POLICY tenant_isolation_policy ON diagnostic_sessions
    USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation_policy ON diagnostic_results
    USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation_policy ON matched_symptoms
    USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation_policy ON differential_diagnoses
    USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation_policy ON required_exams
    USING (organization_id = current_setting('app.current_organization_id')::UUID);

CREATE POLICY tenant_isolation_policy ON organization_treatment_protocols
    USING (organization_id = current_setting('app.current_organization_id')::UUID);