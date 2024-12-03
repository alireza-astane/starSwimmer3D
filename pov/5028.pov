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
    sphere { m*<-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 1 }        
    sphere {  m*<0.5218234738177541,0.2905297000106052,8.294789430354331>, 1 }
    sphere {  m*<2.6177887460625318,-0.030885191795011452,-2.9800509079307553>, 1 }
    sphere {  m*<-1.9227792713742802,2.189017330284023,-2.633262593672493>, 1}
    sphere { m*<-1.6549920503364484,-2.6986746121198744,-2.4437163085099227>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5218234738177541,0.2905297000106052,8.294789430354331>, <-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 0.5 }
    cylinder { m*<2.6177887460625318,-0.030885191795011452,-2.9800509079307553>, <-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 0.5}
    cylinder { m*<-1.9227792713742802,2.189017330284023,-2.633262593672493>, <-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 0.5 }
    cylinder {  m*<-1.6549920503364484,-2.6986746121198744,-2.4437163085099227>, <-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 0.5}

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
    sphere { m*<-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 1 }        
    sphere {  m*<0.5218234738177541,0.2905297000106052,8.294789430354331>, 1 }
    sphere {  m*<2.6177887460625318,-0.030885191795011452,-2.9800509079307553>, 1 }
    sphere {  m*<-1.9227792713742802,2.189017330284023,-2.633262593672493>, 1}
    sphere { m*<-1.6549920503364484,-2.6986746121198744,-2.4437163085099227>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5218234738177541,0.2905297000106052,8.294789430354331>, <-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 0.5 }
    cylinder { m*<2.6177887460625318,-0.030885191795011452,-2.9800509079307553>, <-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 0.5}
    cylinder { m*<-1.9227792713742802,2.189017330284023,-2.633262593672493>, <-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 0.5 }
    cylinder {  m*<-1.6549920503364484,-2.6986746121198744,-2.4437163085099227>, <-0.2994947285908632,-0.13947469591756906,-1.6621449428547375>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    