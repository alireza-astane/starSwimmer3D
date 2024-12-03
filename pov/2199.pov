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
    sphere { m*<1.1248999146970655,0.2667900149673394,0.5309830659358675>, 1 }        
    sphere {  m*<1.3690432823358696,0.287275098652236,3.520961176714829>, 1 }
    sphere {  m*<3.8622904713984063,0.287275098652236,-0.6963210317757871>, 1 }
    sphere {  m*<-3.2203611517898776,7.229395383944566,-2.0382199435534236>, 1}
    sphere { m*<-3.7609799017485623,-7.965881929253955,-2.357188594401655>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3690432823358696,0.287275098652236,3.520961176714829>, <1.1248999146970655,0.2667900149673394,0.5309830659358675>, 0.5 }
    cylinder { m*<3.8622904713984063,0.287275098652236,-0.6963210317757871>, <1.1248999146970655,0.2667900149673394,0.5309830659358675>, 0.5}
    cylinder { m*<-3.2203611517898776,7.229395383944566,-2.0382199435534236>, <1.1248999146970655,0.2667900149673394,0.5309830659358675>, 0.5 }
    cylinder {  m*<-3.7609799017485623,-7.965881929253955,-2.357188594401655>, <1.1248999146970655,0.2667900149673394,0.5309830659358675>, 0.5}

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
    sphere { m*<1.1248999146970655,0.2667900149673394,0.5309830659358675>, 1 }        
    sphere {  m*<1.3690432823358696,0.287275098652236,3.520961176714829>, 1 }
    sphere {  m*<3.8622904713984063,0.287275098652236,-0.6963210317757871>, 1 }
    sphere {  m*<-3.2203611517898776,7.229395383944566,-2.0382199435534236>, 1}
    sphere { m*<-3.7609799017485623,-7.965881929253955,-2.357188594401655>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3690432823358696,0.287275098652236,3.520961176714829>, <1.1248999146970655,0.2667900149673394,0.5309830659358675>, 0.5 }
    cylinder { m*<3.8622904713984063,0.287275098652236,-0.6963210317757871>, <1.1248999146970655,0.2667900149673394,0.5309830659358675>, 0.5}
    cylinder { m*<-3.2203611517898776,7.229395383944566,-2.0382199435534236>, <1.1248999146970655,0.2667900149673394,0.5309830659358675>, 0.5 }
    cylinder {  m*<-3.7609799017485623,-7.965881929253955,-2.357188594401655>, <1.1248999146970655,0.2667900149673394,0.5309830659358675>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    