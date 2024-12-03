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
    sphere { m*<0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 1 }        
    sphere {  m*<0.5823265351131628,-2.5692397393852976e-19,4.003970675006278>, 1 }
    sphere {  m*<7.437331641425071,3.581228696598954e-18,-1.640124539553249>, 1 }
    sphere {  m*<-4.2872286328171585,8.164965809277259,-2.210598506742194>, 1}
    sphere { m*<-4.2872286328171585,-8.164965809277259,-2.2105985067421976>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5823265351131628,-2.5692397393852976e-19,4.003970675006278>, <0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 0.5 }
    cylinder { m*<7.437331641425071,3.581228696598954e-18,-1.640124539553249>, <0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 0.5}
    cylinder { m*<-4.2872286328171585,8.164965809277259,-2.210598506742194>, <0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 0.5 }
    cylinder {  m*<-4.2872286328171585,-8.164965809277259,-2.2105985067421976>, <0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 0.5}

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
    sphere { m*<0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 1 }        
    sphere {  m*<0.5823265351131628,-2.5692397393852976e-19,4.003970675006278>, 1 }
    sphere {  m*<7.437331641425071,3.581228696598954e-18,-1.640124539553249>, 1 }
    sphere {  m*<-4.2872286328171585,8.164965809277259,-2.210598506742194>, 1}
    sphere { m*<-4.2872286328171585,-8.164965809277259,-2.2105985067421976>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5823265351131628,-2.5692397393852976e-19,4.003970675006278>, <0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 0.5 }
    cylinder { m*<7.437331641425071,3.581228696598954e-18,-1.640124539553249>, <0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 0.5}
    cylinder { m*<-4.2872286328171585,8.164965809277259,-2.210598506742194>, <0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 0.5 }
    cylinder {  m*<-4.2872286328171585,-8.164965809277259,-2.2105985067421976>, <0.5079799899902134,-4.480264002384948e-18,1.004889540268341>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    