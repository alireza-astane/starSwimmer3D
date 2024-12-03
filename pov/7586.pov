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
    sphere { m*<-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 1 }        
    sphere {  m*<0.8899018461212568,0.31676144330290734,9.354629759221877>, 1 }
    sphere {  m*<8.257689044444062,0.03166919251064626,-5.216047669852051>, 1 }
    sphere {  m*<-6.6382741492449355,6.554750566131287,-3.725240766670445>, 1}
    sphere { m*<-3.441903767141453,-7.016334578539203,-1.8434667865746215>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8899018461212568,0.31676144330290734,9.354629759221877>, <-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 0.5 }
    cylinder { m*<8.257689044444062,0.03166919251064626,-5.216047669852051>, <-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 0.5}
    cylinder { m*<-6.6382741492449355,6.554750566131287,-3.725240766670445>, <-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 0.5 }
    cylinder {  m*<-3.441903767141453,-7.016334578539203,-1.8434667865746215>, <-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 0.5}

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
    sphere { m*<-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 1 }        
    sphere {  m*<0.8899018461212568,0.31676144330290734,9.354629759221877>, 1 }
    sphere {  m*<8.257689044444062,0.03166919251064626,-5.216047669852051>, 1 }
    sphere {  m*<-6.6382741492449355,6.554750566131287,-3.725240766670445>, 1}
    sphere { m*<-3.441903767141453,-7.016334578539203,-1.8434667865746215>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8899018461212568,0.31676144330290734,9.354629759221877>, <-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 0.5 }
    cylinder { m*<8.257689044444062,0.03166919251064626,-5.216047669852051>, <-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 0.5}
    cylinder { m*<-6.6382741492449355,6.554750566131287,-3.725240766670445>, <-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 0.5 }
    cylinder {  m*<-3.441903767141453,-7.016334578539203,-1.8434667865746215>, <-0.5292656480789049,-0.6731774705770096,-0.49466033781326885>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    