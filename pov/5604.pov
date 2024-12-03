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
    sphere { m*<-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 1 }        
    sphere {  m*<0.18294751689878475,0.2833446145109878,8.588250488602368>, 1 }
    sphere {  m*<5.423487726993677,0.06118025682090211,-4.567408541036399>, 1 }
    sphere {  m*<-2.698633994057454,2.161937373217419,-2.236234069432224>, 1}
    sphere { m*<-2.430846773019623,-2.725754569186478,-2.0466877842696536>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18294751689878475,0.2833446145109878,8.588250488602368>, <-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 0.5 }
    cylinder { m*<5.423487726993677,0.06118025682090211,-4.567408541036399>, <-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 0.5}
    cylinder { m*<-2.698633994057454,2.161937373217419,-2.236234069432224>, <-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 0.5 }
    cylinder {  m*<-2.430846773019623,-2.725754569186478,-2.0466877842696536>, <-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 0.5}

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
    sphere { m*<-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 1 }        
    sphere {  m*<0.18294751689878475,0.2833446145109878,8.588250488602368>, 1 }
    sphere {  m*<5.423487726993677,0.06118025682090211,-4.567408541036399>, 1 }
    sphere {  m*<-2.698633994057454,2.161937373217419,-2.236234069432224>, 1}
    sphere { m*<-2.430846773019623,-2.725754569186478,-2.0466877842696536>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18294751689878475,0.2833446145109878,8.588250488602368>, <-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 0.5 }
    cylinder { m*<5.423487726993677,0.06118025682090211,-4.567408541036399>, <-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 0.5}
    cylinder { m*<-2.698633994057454,2.161937373217419,-2.236234069432224>, <-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 0.5 }
    cylinder {  m*<-2.430846773019623,-2.725754569186478,-2.0466877842696536>, <-1.0409523277207613,-0.1670403180903451,-1.3263189479501007>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    