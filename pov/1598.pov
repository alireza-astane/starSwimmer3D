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
    sphere { m*<0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 1 }        
    sphere {  m*<0.9393863008730443,1.2548307246870494e-18,3.870343612256997>, 1 }
    sphere {  m*<6.194056300808468,5.5615126155475345e-18,-1.2993069131397184>, 1 }
    sphere {  m*<-4.04601958248577,8.164965809277259,-2.251687225128613>, 1}
    sphere { m*<-4.04601958248577,-8.164965809277259,-2.2516872251286166>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9393863008730443,1.2548307246870494e-18,3.870343612256997>, <0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 0.5 }
    cylinder { m*<6.194056300808468,5.5615126155475345e-18,-1.2993069131397184>, <0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 0.5}
    cylinder { m*<-4.04601958248577,8.164965809277259,-2.251687225128613>, <0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 0.5 }
    cylinder {  m*<-4.04601958248577,-8.164965809277259,-2.2516872251286166>, <0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 0.5}

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
    sphere { m*<0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 1 }        
    sphere {  m*<0.9393863008730443,1.2548307246870494e-18,3.870343612256997>, 1 }
    sphere {  m*<6.194056300808468,5.5615126155475345e-18,-1.2993069131397184>, 1 }
    sphere {  m*<-4.04601958248577,8.164965809277259,-2.251687225128613>, 1}
    sphere { m*<-4.04601958248577,-8.164965809277259,-2.2516872251286166>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9393863008730443,1.2548307246870494e-18,3.870343612256997>, <0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 0.5 }
    cylinder { m*<6.194056300808468,5.5615126155475345e-18,-1.2993069131397184>, <0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 0.5}
    cylinder { m*<-4.04601958248577,8.164965809277259,-2.251687225128613>, <0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 0.5 }
    cylinder {  m*<-4.04601958248577,-8.164965809277259,-2.2516872251286166>, <0.8087319470412736,-3.235713287124196e-18,0.8731852370890871>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    