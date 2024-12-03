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
    sphere { m*<0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 1 }        
    sphere {  m*<0.5541093599254802,-6.3668135456057725e-19,4.0139327918866705>, 1 }
    sphere {  m*<7.534551604788623,3.4200757595445736e-18,-1.6656132953553044>, 1 }
    sphere {  m*<-4.307002509547991,8.164965809277259,-2.207242551696454>, 1}
    sphere { m*<-4.307002509547991,-8.164965809277259,-2.2072425516964573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5541093599254802,-6.3668135456057725e-19,4.0139327918866705>, <0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 0.5 }
    cylinder { m*<7.534551604788623,3.4200757595445736e-18,-1.6656132953553044>, <0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 0.5}
    cylinder { m*<-4.307002509547991,8.164965809277259,-2.207242551696454>, <0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 0.5 }
    cylinder {  m*<-4.307002509547991,-8.164965809277259,-2.2072425516964573>, <0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 0.5}

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
    sphere { m*<0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 1 }        
    sphere {  m*<0.5541093599254802,-6.3668135456057725e-19,4.0139327918866705>, 1 }
    sphere {  m*<7.534551604788623,3.4200757595445736e-18,-1.6656132953553044>, 1 }
    sphere {  m*<-4.307002509547991,8.164965809277259,-2.207242551696454>, 1}
    sphere { m*<-4.307002509547991,-8.164965809277259,-2.2072425516964573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5541093599254802,-6.3668135456057725e-19,4.0139327918866705>, <0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 0.5 }
    cylinder { m*<7.534551604788623,3.4200757595445736e-18,-1.6656132953553044>, <0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 0.5}
    cylinder { m*<-4.307002509547991,8.164965809277259,-2.207242551696454>, <0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 0.5 }
    cylinder {  m*<-4.307002509547991,-8.164965809277259,-2.2072425516964573>, <0.4838352411228219,-4.970952811190232e-18,1.0147536228764882>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    