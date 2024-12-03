#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.7872111043643011,0.7989970614798897,0.33131969513625037>, 1 }        
    sphere {  m*<1.0302105949676277,0.8698592891804449,3.3206194139210403>, 1 }
    sphere {  m*<3.523457784030162,0.8698592891804446,-0.8966627945695735>, 1 }
    sphere {  m*<-2.1457806722285944,5.2216821145188534,-1.4028454531749135>, 1}
    sphere { m*<-3.890443530560502,-7.595341610498787,-2.4337428839641326>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0302105949676277,0.8698592891804449,3.3206194139210403>, <0.7872111043643011,0.7989970614798897,0.33131969513625037>, 0.5 }
    cylinder { m*<3.523457784030162,0.8698592891804446,-0.8966627945695735>, <0.7872111043643011,0.7989970614798897,0.33131969513625037>, 0.5}
    cylinder { m*<-2.1457806722285944,5.2216821145188534,-1.4028454531749135>, <0.7872111043643011,0.7989970614798897,0.33131969513625037>, 0.5 }
    cylinder {  m*<-3.890443530560502,-7.595341610498787,-2.4337428839641326>, <0.7872111043643011,0.7989970614798897,0.33131969513625037>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.7872111043643011,0.7989970614798897,0.33131969513625037>, 1 }        
    sphere {  m*<1.0302105949676277,0.8698592891804449,3.3206194139210403>, 1 }
    sphere {  m*<3.523457784030162,0.8698592891804446,-0.8966627945695735>, 1 }
    sphere {  m*<-2.1457806722285944,5.2216821145188534,-1.4028454531749135>, 1}
    sphere { m*<-3.890443530560502,-7.595341610498787,-2.4337428839641326>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0302105949676277,0.8698592891804449,3.3206194139210403>, <0.7872111043643011,0.7989970614798897,0.33131969513625037>, 0.5 }
    cylinder { m*<3.523457784030162,0.8698592891804446,-0.8966627945695735>, <0.7872111043643011,0.7989970614798897,0.33131969513625037>, 0.5}
    cylinder { m*<-2.1457806722285944,5.2216821145188534,-1.4028454531749135>, <0.7872111043643011,0.7989970614798897,0.33131969513625037>, 0.5 }
    cylinder {  m*<-3.890443530560502,-7.595341610498787,-2.4337428839641326>, <0.7872111043643011,0.7989970614798897,0.33131969513625037>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    