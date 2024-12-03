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
    sphere { m*<-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 1 }        
    sphere {  m*<0.16985191853416565,0.20259397969973703,2.8184350431921548>, 1 }
    sphere {  m*<2.663825207798736,0.17591787690578609,-1.3983292533795812>, 1 }
    sphere {  m*<-1.6924985461004178,2.402357845938014,-1.1430654933443667>, 1}
    sphere { m*<-1.9017101059858401,-3.387031904422691,-1.2298892917243975>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16985191853416565,0.20259397969973703,2.8184350431921548>, <-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 0.5 }
    cylinder { m*<2.663825207798736,0.17591787690578609,-1.3983292533795812>, <-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 0.5}
    cylinder { m*<-1.6924985461004178,2.402357845938014,-1.1430654933443667>, <-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 0.5 }
    cylinder {  m*<-1.9017101059858401,-3.387031904422691,-1.2298892917243975>, <-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 0.5}

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
    sphere { m*<-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 1 }        
    sphere {  m*<0.16985191853416565,0.20259397969973703,2.8184350431921548>, 1 }
    sphere {  m*<2.663825207798736,0.17591787690578609,-1.3983292533795812>, 1 }
    sphere {  m*<-1.6924985461004178,2.402357845938014,-1.1430654933443667>, 1}
    sphere { m*<-1.9017101059858401,-3.387031904422691,-1.2298892917243975>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16985191853416565,0.20259397969973703,2.8184350431921548>, <-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 0.5 }
    cylinder { m*<2.663825207798736,0.17591787690578609,-1.3983292533795812>, <-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 0.5}
    cylinder { m*<-1.6924985461004178,2.402357845938014,-1.1430654933443667>, <-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 0.5 }
    cylinder {  m*<-1.9017101059858401,-3.387031904422691,-1.2298892917243975>, <-0.07088318620752587,0.07388390151941171,-0.16911972792839486>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    