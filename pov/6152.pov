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
    sphere { m*<-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 1 }        
    sphere {  m*<-0.03572067953153435,0.16668257997627878,8.902451744250426>, 1 }
    sphere {  m*<7.319630758468439,0.07776230398192152,-5.677041545794932>, 1 }
    sphere {  m*<-3.822500541969144,2.7726856421712425,-2.169471690102216>, 1}
    sphere { m*<-2.8656708568432716,-2.930739005695986,-1.6518769944796614>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.03572067953153435,0.16668257997627878,8.902451744250426>, <-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 0.5 }
    cylinder { m*<7.319630758468439,0.07776230398192152,-5.677041545794932>, <-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 0.5}
    cylinder { m*<-3.822500541969144,2.7726856421712425,-2.169471690102216>, <-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 0.5 }
    cylinder {  m*<-2.8656708568432716,-2.930739005695986,-1.6518769944796614>, <-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 0.5}

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
    sphere { m*<-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 1 }        
    sphere {  m*<-0.03572067953153435,0.16668257997627878,8.902451744250426>, 1 }
    sphere {  m*<7.319630758468439,0.07776230398192152,-5.677041545794932>, 1 }
    sphere {  m*<-3.822500541969144,2.7726856421712425,-2.169471690102216>, 1}
    sphere { m*<-2.8656708568432716,-2.930739005695986,-1.6518769944796614>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.03572067953153435,0.16668257997627878,8.902451744250426>, <-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 0.5 }
    cylinder { m*<7.319630758468439,0.07776230398192152,-5.677041545794932>, <-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 0.5}
    cylinder { m*<-3.822500541969144,2.7726856421712425,-2.169471690102216>, <-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 0.5 }
    cylinder {  m*<-2.8656708568432716,-2.930739005695986,-1.6518769944796614>, <-1.4966005729431018,-0.3484141639412778,-0.9769317618465094>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    