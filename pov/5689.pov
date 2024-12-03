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
    sphere { m*<-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 1 }        
    sphere {  m*<0.12146714374729306,0.28206196038157827,8.642031785254597>, 1 }
    sphere {  m*<5.818100327036448,0.07329250896342354,-4.812293315398911>, 1 }
    sphere {  m*<-2.820488598804093,2.157963224707012,-2.1666636220854145>, 1}
    sphere { m*<-2.5527013777662617,-2.729728717696885,-1.977117336922844>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12146714374729306,0.28206196038157827,8.642031785254597>, <-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 0.5 }
    cylinder { m*<5.818100327036448,0.07329250896342354,-4.812293315398911>, <-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 0.5}
    cylinder { m*<-2.820488598804093,2.157963224707012,-2.1666636220854145>, <-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 0.5 }
    cylinder {  m*<-2.5527013777662617,-2.729728717696885,-1.977117336922844>, <-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 0.5}

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
    sphere { m*<-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 1 }        
    sphere {  m*<0.12146714374729306,0.28206196038157827,8.642031785254597>, 1 }
    sphere {  m*<5.818100327036448,0.07329250896342354,-4.812293315398911>, 1 }
    sphere {  m*<-2.820488598804093,2.157963224707012,-2.1666636220854145>, 1}
    sphere { m*<-2.5527013777662617,-2.729728717696885,-1.977117336922844>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12146714374729306,0.28206196038157827,8.642031785254597>, <-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 0.5 }
    cylinder { m*<5.818100327036448,0.07329250896342354,-4.812293315398911>, <-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 0.5}
    cylinder { m*<-2.820488598804093,2.157963224707012,-2.1666636220854145>, <-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 0.5 }
    cylinder {  m*<-2.5527013777662617,-2.729728717696885,-1.977117336922844>, <-1.1582245550916508,-0.17109690204135164,-1.2653604614424603>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    