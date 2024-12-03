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
    sphere { m*<0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 1 }        
    sphere {  m*<0.63241528086679,-6.175025167739765e-19,3.9860884795374076>, 1 }
    sphere {  m*<7.2644690515531405,3.129088336290039e-18,-1.5944324060906492>, 1 }
    sphere {  m*<-4.252375359760542,8.164965809277259,-2.2165133757481392>, 1}
    sphere { m*<-4.252375359760542,-8.164965809277259,-2.216513375748142>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.63241528086679,-6.175025167739765e-19,3.9860884795374076>, <0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 0.5 }
    cylinder { m*<7.2644690515531405,3.129088336290039e-18,-1.5944324060906492>, <0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 0.5}
    cylinder { m*<-4.252375359760542,8.164965809277259,-2.2165133757481392>, <0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 0.5 }
    cylinder {  m*<-4.252375359760542,-8.164965809277259,-2.216513375748142>, <0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 0.5}

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
    sphere { m*<0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 1 }        
    sphere {  m*<0.63241528086679,-6.175025167739765e-19,3.9860884795374076>, 1 }
    sphere {  m*<7.2644690515531405,3.129088336290039e-18,-1.5944324060906492>, 1 }
    sphere {  m*<-4.252375359760542,8.164965809277259,-2.2165133757481392>, 1}
    sphere { m*<-4.252375359760542,-8.164965809277259,-2.216513375748142>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.63241528086679,-6.175025167739765e-19,3.9860884795374076>, <0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 0.5 }
    cylinder { m*<7.2644690515531405,3.129088336290039e-18,-1.5944324060906492>, <0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 0.5}
    cylinder { m*<-4.252375359760542,8.164965809277259,-2.2165133757481392>, <0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 0.5 }
    cylinder {  m*<-4.252375359760542,-8.164965809277259,-2.216513375748142>, <0.5507096309020972,-4.494336512805576e-18,0.987198527927109>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    