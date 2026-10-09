import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, JSON, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database.connection import Base
from typing import List, Dict, Any, Optional

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    gender: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    birth_date: Mapped[Optional[datetime.date]] = mapped_column(JSON, nullable=True) # stored as JSON or string
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    encounters: Mapped[List["Encounter"]] = relationship(back_populates="patient", cascade="all, delete-orphan")

class Encounter(Base):
    __tablename__ = "encounters"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    start_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    patient: Mapped["Patient"] = relationship(back_populates="encounters")
    observations: Mapped[List["Observation"]] = relationship(back_populates="encounter", cascade="all, delete-orphan")
    medication_requests: Mapped[List["MedicationRequest"]] = relationship(back_populates="encounter", cascade="all, delete-orphan")
    procedures: Mapped[List["Procedure"]] = relationship(back_populates="encounter", cascade="all, delete-orphan")
    document_references: Mapped[List["DocumentReference"]] = relationship(back_populates="encounter", cascade="all, delete-orphan")

class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    encounter_id: Mapped[Optional[str]] = mapped_column(ForeignKey("encounters.id", ondelete="CASCADE"), nullable=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    display: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    value_quantity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    value_string: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    effective_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    encounter: Mapped[Optional["Encounter"]] = relationship(back_populates="observations")

class MedicationRequest(Base):
    __tablename__ = "medication_requests"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    encounter_id: Mapped[Optional[str]] = mapped_column(ForeignKey("encounters.id", ondelete="CASCADE"), nullable=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    medication_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    authored_on: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    encounter: Mapped[Optional["Encounter"]] = relationship(back_populates="medication_requests")

class Procedure(Base):
    __tablename__ = "procedures"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    encounter_id: Mapped[Optional[str]] = mapped_column(ForeignKey("encounters.id", ondelete="CASCADE"), nullable=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    display: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    performed_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    encounter: Mapped[Optional["Encounter"]] = relationship(back_populates="procedures")

class DocumentReference(Base):
    __tablename__ = "document_references"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    encounter_id: Mapped[Optional[str]] = mapped_column(ForeignKey("encounters.id", ondelete="CASCADE"), nullable=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    type_code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    type_display: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    content_text: Mapped[str] = mapped_column(Text) # anonymized note text
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    encounter: Mapped[Optional["Encounter"]] = relationship(back_populates="document_references")

class DeIdAuditLog(Base):
    __tablename__ = "deid_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    resource_type: Mapped[str] = mapped_column(String)
    resource_id: Mapped[str] = mapped_column(String)
    original_length: Mapped[int] = mapped_column(Integer)
    anonymized_length: Mapped[int] = mapped_column(Integer)
    scrubbed_entities: Mapped[List[str]] = mapped_column(JSON) # e.g. ["NAME", "PHONE"]
    status: Mapped[str] = mapped_column(String)
