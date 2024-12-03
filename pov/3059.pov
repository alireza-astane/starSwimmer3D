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
    sphere { m*<0.4665051797063491,1.0897395618381527,0.1422397000663229>, 1 }        
    sphere {  m*<0.7072402844480407,1.218449640018478,3.1297944711868726>, 1 }
    sphere {  m*<3.2012135737126055,1.1917735372245273,-1.0869698253848616>, 1 }
    sphere {  m*<-1.15511018018654,3.4182135062567536,-0.8317060653496476>, 1}
    sphere { m*<-3.85671934370414,-7.082696578394899,-2.3626093068761804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7072402844480407,1.218449640018478,3.1297944711868726>, <0.4665051797063491,1.0897395618381527,0.1422397000663229>, 0.5 }
    cylinder { m*<3.2012135737126055,1.1917735372245273,-1.0869698253848616>, <0.4665051797063491,1.0897395618381527,0.1422397000663229>, 0.5}
    cylinder { m*<-1.15511018018654,3.4182135062567536,-0.8317060653496476>, <0.4665051797063491,1.0897395618381527,0.1422397000663229>, 0.5 }
    cylinder {  m*<-3.85671934370414,-7.082696578394899,-2.3626093068761804>, <0.4665051797063491,1.0897395618381527,0.1422397000663229>, 0.5}

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
    sphere { m*<0.4665051797063491,1.0897395618381527,0.1422397000663229>, 1 }        
    sphere {  m*<0.7072402844480407,1.218449640018478,3.1297944711868726>, 1 }
    sphere {  m*<3.2012135737126055,1.1917735372245273,-1.0869698253848616>, 1 }
    sphere {  m*<-1.15511018018654,3.4182135062567536,-0.8317060653496476>, 1}
    sphere { m*<-3.85671934370414,-7.082696578394899,-2.3626093068761804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7072402844480407,1.218449640018478,3.1297944711868726>, <0.4665051797063491,1.0897395618381527,0.1422397000663229>, 0.5 }
    cylinder { m*<3.2012135737126055,1.1917735372245273,-1.0869698253848616>, <0.4665051797063491,1.0897395618381527,0.1422397000663229>, 0.5}
    cylinder { m*<-1.15511018018654,3.4182135062567536,-0.8317060653496476>, <0.4665051797063491,1.0897395618381527,0.1422397000663229>, 0.5 }
    cylinder {  m*<-3.85671934370414,-7.082696578394899,-2.3626093068761804>, <0.4665051797063491,1.0897395618381527,0.1422397000663229>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    