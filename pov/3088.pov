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
    sphere { m*<0.44468301413937245,1.048487886480561,0.1295960750466685>, 1 }        
    sphere {  m*<0.6854181188810641,1.1771979646608863,3.1171508461672186>, 1 }
    sphere {  m*<3.179391408145628,1.1505218618669353,-1.099613450404516>, 1 }
    sphere {  m*<-1.1769323457535168,3.3769618308991625,-0.8443496903693023>, 1}
    sphere { m*<-3.7869324718752564,-6.950774502555487,-2.322175233732023>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6854181188810641,1.1771979646608863,3.1171508461672186>, <0.44468301413937245,1.048487886480561,0.1295960750466685>, 0.5 }
    cylinder { m*<3.179391408145628,1.1505218618669353,-1.099613450404516>, <0.44468301413937245,1.048487886480561,0.1295960750466685>, 0.5}
    cylinder { m*<-1.1769323457535168,3.3769618308991625,-0.8443496903693023>, <0.44468301413937245,1.048487886480561,0.1295960750466685>, 0.5 }
    cylinder {  m*<-3.7869324718752564,-6.950774502555487,-2.322175233732023>, <0.44468301413937245,1.048487886480561,0.1295960750466685>, 0.5}

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
    sphere { m*<0.44468301413937245,1.048487886480561,0.1295960750466685>, 1 }        
    sphere {  m*<0.6854181188810641,1.1771979646608863,3.1171508461672186>, 1 }
    sphere {  m*<3.179391408145628,1.1505218618669353,-1.099613450404516>, 1 }
    sphere {  m*<-1.1769323457535168,3.3769618308991625,-0.8443496903693023>, 1}
    sphere { m*<-3.7869324718752564,-6.950774502555487,-2.322175233732023>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6854181188810641,1.1771979646608863,3.1171508461672186>, <0.44468301413937245,1.048487886480561,0.1295960750466685>, 0.5 }
    cylinder { m*<3.179391408145628,1.1505218618669353,-1.099613450404516>, <0.44468301413937245,1.048487886480561,0.1295960750466685>, 0.5}
    cylinder { m*<-1.1769323457535168,3.3769618308991625,-0.8443496903693023>, <0.44468301413937245,1.048487886480561,0.1295960750466685>, 0.5 }
    cylinder {  m*<-3.7869324718752564,-6.950774502555487,-2.322175233732023>, <0.44468301413937245,1.048487886480561,0.1295960750466685>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    