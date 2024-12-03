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
    sphere { m*<-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 1 }        
    sphere {  m*<-0.10216054443254752,0.22381246667580051,8.868582271711945>, 1 }
    sphere {  m*<7.25319089356742,0.13489219068144265,-5.710911018333418>, 1 }
    sphere {  m*<-3.4203997926661804,2.314678935780812,-1.9638042739817547>, 1}
    sphere { m*<-2.9613730523218313,-2.7937160116384945,-1.7009981710203825>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.10216054443254752,0.22381246667580051,8.868582271711945>, <-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 0.5 }
    cylinder { m*<7.25319089356742,0.13489219068144265,-5.710911018333418>, <-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 0.5}
    cylinder { m*<-3.4203997926661804,2.314678935780812,-1.9638042739817547>, <-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 0.5 }
    cylinder {  m*<-2.9613730523218313,-2.7937160116384945,-1.7009981710203825>, <-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 0.5}

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
    sphere { m*<-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 1 }        
    sphere {  m*<-0.10216054443254752,0.22381246667580051,8.868582271711945>, 1 }
    sphere {  m*<7.25319089356742,0.13489219068144265,-5.710911018333418>, 1 }
    sphere {  m*<-3.4203997926661804,2.314678935780812,-1.9638042739817547>, 1}
    sphere { m*<-2.9613730523218313,-2.7937160116384945,-1.7009981710203825>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.10216054443254752,0.22381246667580051,8.868582271711945>, <-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 0.5 }
    cylinder { m*<7.25319089356742,0.13489219068144265,-5.710911018333418>, <-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 0.5}
    cylinder { m*<-3.4203997926661804,2.314678935780812,-1.9638042739817547>, <-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 0.5 }
    cylinder {  m*<-2.9613730523218313,-2.7937160116384945,-1.7009981710203825>, <-1.5671811369932405,-0.22825994665533583,-1.013275445534377>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    