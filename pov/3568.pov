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
    sphere { m*<0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 1 }        
    sphere {  m*<0.34310781102539023,0.5301094047340651,2.9188184139617173>, 1 }
    sphere {  m*<2.8370811002899567,0.5034333019401139,-1.2979458826100179>, 1 }
    sphere {  m*<-1.5192426536091923,2.72987327097234,-1.0426821225748033>, 1}
    sphere { m*<-2.612955264213258,-4.731538904738758,-1.6419802480537182>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.34310781102539023,0.5301094047340651,2.9188184139617173>, <0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 0.5 }
    cylinder { m*<2.8370811002899567,0.5034333019401139,-1.2979458826100179>, <0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 0.5}
    cylinder { m*<-1.5192426536091923,2.72987327097234,-1.0426821225748033>, <0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 0.5 }
    cylinder {  m*<-2.612955264213258,-4.731538904738758,-1.6419802480537182>, <0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 0.5}

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
    sphere { m*<0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 1 }        
    sphere {  m*<0.34310781102539023,0.5301094047340651,2.9188184139617173>, 1 }
    sphere {  m*<2.8370811002899567,0.5034333019401139,-1.2979458826100179>, 1 }
    sphere {  m*<-1.5192426536091923,2.72987327097234,-1.0426821225748033>, 1}
    sphere { m*<-2.612955264213258,-4.731538904738758,-1.6419802480537182>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.34310781102539023,0.5301094047340651,2.9188184139617173>, <0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 0.5 }
    cylinder { m*<2.8370811002899567,0.5034333019401139,-1.2979458826100179>, <0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 0.5}
    cylinder { m*<-1.5192426536091923,2.72987327097234,-1.0426821225748033>, <0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 0.5 }
    cylinder {  m*<-2.612955264213258,-4.731538904738758,-1.6419802480537182>, <0.10237270628369877,0.40139932655373967,-0.06873635715883206>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    