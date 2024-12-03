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
    sphere { m*<0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 1 }        
    sphere {  m*<0.407913090110354,-3.0540197724388396e-18,4.064332665551482>, 1 }
    sphere {  m*<8.036680239139562,2.626879348442419e-18,-1.7950413715192404>, 1 }
    sphere {  m*<-4.411017352660104,8.164965809277259,-2.1895709147164615>, 1}
    sphere { m*<-4.411017352660104,-8.164965809277259,-2.189570914716464>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.407913090110354,-3.0540197724388396e-18,4.064332665551482>, <0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 0.5 }
    cylinder { m*<8.036680239139562,2.626879348442419e-18,-1.7950413715192404>, <0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 0.5}
    cylinder { m*<-4.411017352660104,8.164965809277259,-2.1895709147164615>, <0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 0.5 }
    cylinder {  m*<-4.411017352660104,-8.164965809277259,-2.189570914716464>, <0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 0.5}

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
    sphere { m*<0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 1 }        
    sphere {  m*<0.407913090110354,-3.0540197724388396e-18,4.064332665551482>, 1 }
    sphere {  m*<8.036680239139562,2.626879348442419e-18,-1.7950413715192404>, 1 }
    sphere {  m*<-4.411017352660104,8.164965809277259,-2.1895709147164615>, 1}
    sphere { m*<-4.411017352660104,-8.164965809277259,-2.189570914716464>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.407913090110354,-3.0540197724388396e-18,4.064332665551482>, <0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 0.5 }
    cylinder { m*<8.036680239139562,2.626879348442419e-18,-1.7950413715192404>, <0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 0.5}
    cylinder { m*<-4.411017352660104,8.164965809277259,-2.1895709147164615>, <0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 0.5 }
    cylinder {  m*<-4.411017352660104,-8.164965809277259,-2.189570914716464>, <0.3579217670715382,-5.252282895324708e-18,1.0647475975074752>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    