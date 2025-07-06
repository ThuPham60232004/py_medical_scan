from fastapi import APIRouter, Depends, HTTPException,status
from app.controller.location_controller import search_pharmacies
# from app.models.userModel import Location

router = APIRouter() 

@router.get("/pharmacy", response_model=list)
async def get_pharmacy(lat:float,lng:float):
    """
    Tìm kiếm các hiệu thuốc gần vị trí hiện tại.
    """
    try:
        pharmacies = await search_pharmacies(lat,lng)
        if not pharmacies:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy hiệu thuốc nào gần đây.")
        return pharmacies
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
