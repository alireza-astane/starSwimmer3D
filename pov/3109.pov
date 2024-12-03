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
    sphere { m*<0.4289439301998593,1.0187354048074673,0.12047694913437988>, 1 }        
    sphere {  m*<0.669679034941551,1.147445482987793,3.108031720254931>, 1 }
    sphere {  m*<3.1636523242061156,1.1207693801938419,-1.1087325763168043>, 1 }
    sphere {  m*<-1.1926714296930303,3.347209349226069,-0.8534688162815905>, 1}
    sphere { m*<-3.7363339770108723,-6.855125302327301,-2.2928587851118953>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.669679034941551,1.147445482987793,3.108031720254931>, <0.4289439301998593,1.0187354048074673,0.12047694913437988>, 0.5 }
    cylinder { m*<3.1636523242061156,1.1207693801938419,-1.1087325763168043>, <0.4289439301998593,1.0187354048074673,0.12047694913437988>, 0.5}
    cylinder { m*<-1.1926714296930303,3.347209349226069,-0.8534688162815905>, <0.4289439301998593,1.0187354048074673,0.12047694913437988>, 0.5 }
    cylinder {  m*<-3.7363339770108723,-6.855125302327301,-2.2928587851118953>, <0.4289439301998593,1.0187354048074673,0.12047694913437988>, 0.5}

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
    sphere { m*<0.4289439301998593,1.0187354048074673,0.12047694913437988>, 1 }        
    sphere {  m*<0.669679034941551,1.147445482987793,3.108031720254931>, 1 }
    sphere {  m*<3.1636523242061156,1.1207693801938419,-1.1087325763168043>, 1 }
    sphere {  m*<-1.1926714296930303,3.347209349226069,-0.8534688162815905>, 1}
    sphere { m*<-3.7363339770108723,-6.855125302327301,-2.2928587851118953>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.669679034941551,1.147445482987793,3.108031720254931>, <0.4289439301998593,1.0187354048074673,0.12047694913437988>, 0.5 }
    cylinder { m*<3.1636523242061156,1.1207693801938419,-1.1087325763168043>, <0.4289439301998593,1.0187354048074673,0.12047694913437988>, 0.5}
    cylinder { m*<-1.1926714296930303,3.347209349226069,-0.8534688162815905>, <0.4289439301998593,1.0187354048074673,0.12047694913437988>, 0.5 }
    cylinder {  m*<-3.7363339770108723,-6.855125302327301,-2.2928587851118953>, <0.4289439301998593,1.0187354048074673,0.12047694913437988>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    