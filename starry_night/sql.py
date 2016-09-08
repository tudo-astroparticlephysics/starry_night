from sqlalchemy import Column, Integer, String, VARCHAR, DateTime, Float, ForeignKey, create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from IPython import embed


Base = declarative_base()

class SqlEntry(Base):
    '''
    Normal data entry for SQL database
    '''
    __tablename__ = 'global'
    timestamp = Column(DateTime, primary_key=True)
    hashsum = Column(VARCHAR(40))
    sunAlt = Column(Float)
    moonAlt = Column(Float)
    moonPhase = Column(Float)
    brightnessMean = Column(Float)
    brightnessStd = Column(Float)


class SqlPoiEntry(Base):
    '''
    Entry for a point of interest object
    '''
    __tablename__ = 'local'
    entry = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(ForeignKey('global.timestamp'))
    ID = Column(Integer, nullable=False)
    starPerc = Column(Float)


def writeSQL(config, data):
    engine = create_engine(config['SQL']['connection'])
    sMaker = sessionmaker(bind=engine)
    session = sMaker()
    
    for i, poi in data['points_of_interest'].iterrows():
        session.add(SqlPoiEntry(timestamp=data['timestamp'], ID=poi.ID, starPerc=poi.starPercentage))
    session.add(SqlEntry(
                    timestamp=data['timestamp'],
                    hashsum=data['hash'],
                    sunAlt=data['sun_alt'],
                    moonAlt=data['moon_alt'],
                    moonPhase=data['moon_phase'],
                    brightnessMean=data['brightness_mean'].item(),
                    brightnessStd=data['brightness_std'].item(),
                )
    )

    Base.metadata.create_all(engine)
    session.commit()
